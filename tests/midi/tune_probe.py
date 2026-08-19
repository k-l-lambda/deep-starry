#!/usr/bin/env python3
'''Measure the numbers that `data.batch_size` / `trainer.epoch_size` must be derived from.

The tuning rule is measure-first: batch_size is bounded by GPU memory at the WORST-CASE sequence
length, and epoch_size is set by balancing a train epoch's wall time against a full val pass. Both
depend on the architecture and on the feeder's real output, so neither transfers between configs — an
l8d512 measurement says nothing about l16d256, and a synthetic fixed-length sweep says nothing about a
corpus whose batch cost is set by its LONGEST row.

Three sections, each answering one question:

  1. memory   — how much does a step cost at max_tokens, and where is the OOM wall?
  2. step     — what is the steady-state train step time, feeder included?
  3. val      — how long is one full pass over the val split?

Section 1 pads to the config's `max_tokens` cap deliberately. A sweep on median-length batches would
report a ceiling the tail then breaks: a batch is sized by its longest row, and attention is O(T^2).

Usage
    python3 tests/midi/tune_probe.py --config configs/<name>.local.yaml \\
        [--sweep 16,20,32,64] [--steps 12] [--val-batches 0] [--device cuda:1]

`--val-batches N` times N batches and extrapolates the full pass (0 = the whole split, which is what
the final number should come from). `--device` matters on a shared node: pick a card with headroom.
'''

import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from starry.utils.config import Configuration			# noqa: E402
from starry.utils.model_factory import loadModel		# noqa: E402
from starry.utils.dataset_factory import loadDataset	# noqa: E402


DATA_DIR = os.environ.get('DATA_DIR')		# same source train.py / trainDist.py read


def build (config_path, device, data_dir=None, batch_size=None):
	'''Volatile so probing never publishes a run directory, and the vocabulary still gets pinned.

	The val loader is built SEPARATELY at `trainer.val_batch_size`, because that is what the distributed
	validator uses — timing the val pass at `data.batch_size` would measure a pass that never happens.

	`batch_size` overrides `data.batch_size` for the TRAIN loader only, so a candidate can be measured on
	real batches without editing the config first. It is written back into the in-memory config so
	section 4 balances against the size actually measured rather than the file's.
	'''
	root = data_dir if data_dir is not None else (DATA_DIR or '.')
	config = Configuration.createOrLoad(config_path, volatile=True)
	if batch_size:
		config['data.batch_size'] = batch_size
	train, _ = loadDataset(config, data_dir=root, device=device,
		batch_size=batch_size or config['data.batch_size'])
	val_bs = config['trainer.val_batch_size'] or config['data.batch_size']
	_, val = loadDataset(config, data_dir=root, device=device, batch_size=val_bs)
	model = loadModel(config['model'], postfix='Loss').to(device)
	return config, train, val, model, val_bs


def optimizer_for (model):
	return torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9,
		weight_decay=0.01)


# Which batch keys carry a sequence axis, per pack. 'split' has TWO independent axes and the cap is
# JOINT over them, which is the whole reason this table exists rather than a single T.
_SPLIT_SOURCE = ('source_ids', 'source_masks', 'source_position_ids')
_SPLIT_TARGET = ('decoder_input_ids', 'decoder_masks', 'decoder_position_ids', 'labels', 'target_mask')


def _seq_len (batch):
	"""Assembled length: the quantity `max_tokens` actually bounds.

	`Seq2Seq2` applies max_tokens to `len(case['ids'])` — the assembled source+<sep>+target sequence —
	BEFORE `pack: split` cuts it at the recorded <sep>. So a split batch's cap is joint over its two
	axes, and the flat batch's single axis already IS the assembled length.
	"""
	if 'input_ids' in batch:
		return int(batch['input_ids'].shape[1])
	return int(batch['source_ids'].shape[1]) + int(batch['decoder_input_ids'].shape[1])


def _pad_axis (batch, keys, target_len, pad_id):
	out = dict(batch)
	for key in keys:
		value = batch.get(key)
		if not torch.is_tensor(value) or value.dim() != 2 or value.shape[1] >= target_len:
			continue
		pad = target_len - value.shape[1]
		fill = pad_id if key in ('input_ids', 'source_ids', 'decoder_input_ids') else 0
		if key.endswith('position_ids'):
			# positions must stay monotone, so continue the run rather than zero it
			tail = value[:, -1:] + torch.arange(1, pad + 1, device=value.device)
			out[key] = torch.cat((value, tail), dim=1)
		else:
			out[key] = torch.cat((value,
				torch.full((value.shape[0], pad), fill, dtype=value.dtype, device=value.device)), dim=1)
	return out


def pad_to (batch, T, pad_id, split_bias=0.75):
	"""Stretch a real batch to the worst case the cap allows.

	Flat: one axis, padded to T.

	Split: the cap is JOINT, so padding both axes to T would measure ~2T tokens and report a ceiling
	the run can never hit. Attention cost is O(S^2) + O(T^2) + O(S*T); with S + T = C that is
	C^2 - S*T, i.e. MAXIMISED at the extremes rather than at an even split. The adverse extreme is the
	decoder-heavy one whenever the decoder stack is the deeper of the two, since the per-layer linear
	terms bill against each axis by its own layer count. `split_bias` is the target's share of the cap.
	"""
	if 'input_ids' in batch:
		return _pad_axis(batch, tuple(batch.keys()), T, pad_id)
	tgt = max(1, int(round(T * split_bias)))
	src = max(1, T - tgt)
	out = _pad_axis(batch, _SPLIT_SOURCE, src, pad_id)
	return _pad_axis(out, _SPLIT_TARGET, tgt, pad_id)


def one_step (model, opt, batch):
	opt.zero_grad(set_to_none=True)
	loss, _ = model(batch)
	loss.backward()
	opt.step()
	return float(loss.detach())


def cycle (loader):
	'''Endless batches, mirroring the trainer's `infiniteTraverse`.

	A plain DataLoader is finite, so on a small corpus the step probe would StopIteration mid-measure
	instead of measuring. The production corpus is large enough that this never fires there, which is
	exactly why it has to be handled here rather than noticed later.
	'''
	while True:
		for batch in loader:
			yield batch


def section_memory (config, train, model, device, sizes):
	'''Peak memory and step time per candidate batch_size, at the config's max_tokens cap.'''
	cap = int((config['data.args'] or {}).get('max_tokens') or 2048)
	pad_id = getattr(model, 'pad_id', 0)
	total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
	it0 = cycle(train)
	probe0 = next(it0)
	if 'input_ids' in probe0:
		shape_note = f'T = {cap}'
	else:
		tgt = max(1, int(round(cap * 0.75)))
		shape_note = f'source {cap - tgt} + target {tgt} = {cap} JOINT (decoder-heavy extreme)'
	print(f'\n== 1. memory at worst case, {shape_note} (card {total:.1f} GB)')
	print('  %4s %11s %11s %7s %9s %11s' % ('bs', 'allocated', 'reserved', 'card%', 'step', 'samples/s'))

	it = cycle(train)
	base = next(it)
	rows = []
	for bs in sizes:
		batch = {k: (v[:1].repeat(bs, *([1] * (v.dim() - 1))) if torch.is_tensor(v) else v)
			for k, v in base.items()}
		batch = pad_to(batch, cap, pad_id)
		opt = optimizer_for(model)
		try:
			torch.cuda.empty_cache()
			torch.cuda.reset_peak_memory_stats(device)
			for _ in range(2):							# warm up the allocator and autotuner
				one_step(model, opt, batch)
			torch.cuda.synchronize(device)
			t0 = time.time()
			for _ in range(3):
				one_step(model, opt, batch)
			torch.cuda.synchronize(device)
			step = (time.time() - t0) / 3
			alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 3
			res = torch.cuda.max_memory_reserved(device) / 1024 ** 3
			print('  %4d %10.2fG %10.2fG %6.1f%% %8.0fms %11.1f' % (
				bs, alloc, res, 100 * res / total, step * 1000, bs / step))
			rows.append((bs, alloc, res, step))
		except torch.OutOfMemoryError:
			print('  %4d %s' % (bs, 'OOM'))
		finally:
			del opt
			torch.cuda.empty_cache()
	return rows


def section_step (train, model, device, steps):
	'''Steady-state FEEDER-INCLUSIVE step time: what an epoch is actually made of.'''
	print(f'\n== 2. train step, feeder included ({steps} steps after 3 warmup)')
	opt = optimizer_for(model)
	it = cycle(train)
	for _ in range(3):
		one_step(model, opt, next(it))
	torch.cuda.synchronize(device)

	times, lengths = [], []
	t0 = time.time()
	for _ in range(steps):
		s = time.time()
		batch = next(it)
		one_step(model, opt, batch)
		torch.cuda.synchronize(device)
		times.append(time.time() - s)
		lengths.append(_seq_len(batch))
	wall = time.time() - t0
	times.sort()
	median = times[len(times) // 2]
	print('  median %.0f ms   mean %.0f ms   min %.0f ms   max %.0f ms' % (
		median * 1000, wall / steps * 1000, times[0] * 1000, times[-1] * 1000))
	print('  T over these steps: median %d  max %d' % (sorted(lengths)[len(lengths) // 2], max(lengths)))
	return median, wall / steps


def section_val (config, val, model, device, limit):
	'''Full val pass time — the fixed cost a train epoch is balanced against.'''
	n = len(val) if hasattr(val, '__len__') else None
	print(f'\n== 3. val pass ({"all" if not limit else limit} batches, {n} reported)')
	model.eval()
	done = 0
	torch.cuda.synchronize(device)
	t0 = time.time()
	with torch.no_grad():
		for batch in val:
			model(batch)
			done += 1
			if limit and done >= limit:
				break
	torch.cuda.synchronize(device)
	elapsed = time.time() - t0
	model.train()
	per = elapsed / max(done, 1)
	print('  %d batches in %.1f s   %.0f ms/batch   %.2f it/s' % (done, elapsed, per * 1000, done / elapsed))
	full = elapsed if not limit or (n and done >= n) else (per * n if n else None)
	if full is not None and full != elapsed:
		print('  extrapolated full pass over %d batches: %.1f s' % (n, full))
	return per, full


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', required=True)
	ap.add_argument('--sweep', default='', help='comma-separated batch sizes; empty skips section 1')
	ap.add_argument('--steps', type=int, default=12)
	ap.add_argument('--val-batches', type=int, default=0, help='0 = full pass')
	ap.add_argument('--device', default='cuda:0')
	ap.add_argument('--data-dir', default=None, help='overrides $DATA_DIR; ignored for an absolute root')
	ap.add_argument('--batch-size', type=int, default=None,
		help='override data.batch_size for the train loader, to measure a candidate before adopting it')
	ap.add_argument('--skip-val', action='store_true')
	args = ap.parse_args()

	device = torch.device(args.device)
	torch.cuda.set_device(device)
	config, train, val, model, val_bs = build(args.config, device, args.data_dir, args.batch_size)

	print('config      : %s' % args.config)
	print('id          : %s' % config.id)
	print('arch        : d_model %s  n_layer %s  n_head %s  vocab %s' % (
		config['model.args.d_model'], config['model.args.n_layer'], config['model.args.n_head'],
		config['model.args.vocab_size']))
	print('params      : %.2fM' % (sum(p.numel() for p in model.deducer.parameters()) / 1e6))
	print('batch_size  : %s   val_batch_size: %s (val loader built at %s)' % (
		config['data.batch_size'], config['trainer.val_batch_size'], val_bs))
	print('device      : %s (%s)' % (device, torch.cuda.get_device_name(device)))

	if args.sweep:
		section_memory(config, train, model, device, [int(x) for x in args.sweep.split(',')])

	median, mean = section_step(train, model, device, args.steps)

	if not args.skip_val:
		per, full = section_val(config, val, model, device, args.val_batches)
		if full:
			bs = int(config['data.batch_size'])
			steps = max(1, round(full / median))			# steps whose wall time matches one val pass
			balanced = steps * bs
			print('\n== 4. epoch_size')
			print('  balancing val: %d steps (%.1f s val / %.3f s step) -> epoch_size %d, train epoch %.0f s'
				% (steps, full, median, balanced, steps * median))
			if steps * median < 60:
				print('  DEGENERATE: the val pass is too short to balance against — a train epoch would be')
				print('  under a minute, so validation dominates. Enlarge the val split before using this.')
			# The operative constraint is usually a target epoch WALL TIME, not parity with val.
			for target in (120, 300):
				s = max(1, round(target / median))
				print('  for a %3d s train epoch: %4d steps -> epoch_size %6d  (val/train ratio %.2f)'
					% (target, s, s * bs, full / (s * median)))


if __name__ == '__main__':
	main()
