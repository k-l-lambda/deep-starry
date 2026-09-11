'''Checks MidiTranslator's KV-cache decode against the recompute path, and times both.

Two claims are made here and each needs its own evidence:

  EQUIVALENCE  cached and uncached decoding produce the same tokens, and logits agreeing to float32
               noise, while the sequence stays inside max_seq_len. Token equality alone is too weak
               to trust — greedy argmax hides a lot of drift — so the logits are compared directly.
  SPEEDUP      wall time per generated token, measured on the real shape the sliding translator
               runs at (prefix ~1200, ~800 new tokens), not on a toy length where per-call overhead
               dominates and the cache looks worse than it is.

Deliberately NOT asserted: equality past a cache trim. KVDecoder._trim documents why the two are
different functions there, and this file measures that divergence rather than pretending it away.

Usage:
	python3 tests/midi/kv_cache_check.py                    # random-init, fast, no checkpoint
	python3 tests/midi/kv_cache_check.py --run <dir>        # a real trained model
'''

import argparse
import os
import sys
import time

import torch

sys.path.append(os.getcwd())

from starry.midi.models.midiTranslator import MidiTranslator, KVDecoder


def build (run, checkpoint, device):
	'''A real run's model when given, else a small random-init one.'''
	if not run:
		model = MidiTranslator(vocab_size=838, d_model=512, n_layer=8, n_head=8, max_seq_len=4096,
			dropout=0.0)
		return model.to(device).eval(), 'random-init d512/l8'
	from starry.utils.config import Configuration
	from starry.utils.model_factory import loadModel
	config = Configuration.createOrLoad(run, volatile=True)
	model = loadModel(config['model'], imports=config['imports'])
	if checkpoint is None:
		checkpoint = os.path.join(run, 'latest.chkpt')
	blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
	state = blob['model'] if isinstance(blob, dict) and 'model' in blob else blob
	model.load_state_dict(state, strict=False)
	return model.to(device).eval(), os.path.basename(run)


def positions_absolute (n_prefix):
	'''The 'absolute'/'sep' shape: a negative source run, <sep> at -1, then the target from 0.

	Uses the axis the trained runs actually use, since the whole point of separating cache_position
	from position_ids is that these values are negative.
	'''
	n_source = n_prefix - 1
	return list(range(-n_source, 0)) + [0]


def check_equivalence (model, device, n_prefix, n_new, vocab, seed):
	torch.manual_seed(seed)
	prefix = torch.randint(6, vocab, (n_prefix,), device=device)
	pos = torch.tensor(positions_absolute(n_prefix), dtype=torch.long, device=device)

	# step-by-step logits from both routes, so a divergence is located rather than just detected
	uncached, cached = [], []
	ids = prefix.clone().unsqueeze(0)
	p = pos.clone().unsqueeze(0)
	with torch.no_grad():
		decoder = KVDecoder(model)
		cached.append(decoder.prefill(prefix.tolist(), pos.tolist(), device=device))
		uncached.append(model(ids, None, p)[0, -1, :])
		nxt = int(uncached[0].argmax().item())
		for _ in range(n_new):
			ids = torch.cat((ids, torch.tensor([[nxt]], device=device)), dim=1)
			p = torch.cat((p, p[:, -1:] + 1), dim=1)
			uncached.append(model(ids, None, p)[0, -1, :])
			cached.append(decoder.step(nxt, int(p[0, -1].item()), device=device))
			nxt = int(uncached[-1].argmax().item())

	u, c = torch.stack(uncached), torch.stack(cached)
	diff = (u - c).abs().max().item()
	agree = int((u.argmax(-1) == c.argmax(-1)).sum())
	return diff, agree, u.shape[0]


def time_route (model, device, n_prefix, n_new, vocab, seed, use_cache):
	torch.manual_seed(seed)
	prefix = torch.randint(6, vocab, (n_prefix,), device=device)
	pos = torch.tensor(positions_absolute(n_prefix), dtype=torch.long, device=device)
	if device.type == 'cuda':
		torch.cuda.synchronize()
	start = time.time()
	with torch.no_grad():
		# eos_id far outside the vocabulary so neither route stops early: the two must decode the
		# SAME number of tokens or the timing compares different amounts of work.
		out = model.generate(prefix, max_new_tokens=n_new, eos_id=-1, temperature=0.0,
			position_ids=pos, use_cache=use_cache)
	if device.type == 'cuda':
		torch.cuda.synchronize()
	return time.time() - start, out


def trim_probe (device, seed):
	'''Measure the cached-vs-uncached divergence ACROSS a cache trim, which is expected.

	KVDecoder._trim cites these numbers. They are produced here rather than asserted because the two
	routes compute different functions past the trim: the cached one keeps hidden states derived from
	a longer history, the recompute one rebuilds them from the cropped window. What matters is the
	SIZE of it, and that it reaches the emitted tokens rather than staying in the low bits.

	Uses a small max_seq_len so a trim happens within a few dozen steps.
	'''
	torch.manual_seed(seed)
	maxlen, prefix_len, n_new = 32, 20, 40
	model = MidiTranslator(vocab_size=64, d_model=64, n_layer=2, n_head=4, max_seq_len=maxlen,
		dropout=0.0).to(device).eval()
	ids = torch.randint(6, 64, (prefix_len + n_new,), device=device)
	pos = torch.arange(-prefix_len + 1, n_new + 1, device=device)

	decoder = KVDecoder(model)
	cached = [decoder.prefill(ids[:prefix_len].tolist(), pos[:prefix_len].tolist(), device=device)]
	uncached = []
	with torch.no_grad():
		uncached.append(model(ids[:prefix_len].unsqueeze(0), None, pos[:prefix_len].unsqueeze(0))[0, -1])
		for i in range(prefix_len, prefix_len + n_new):
			cached.append(decoder.step(int(ids[i]), int(pos[i]), device=device))
			uncached.append(model(ids[:i + 1][-maxlen:].unsqueeze(0), None,
				pos[:i + 1][-maxlen:].unsqueeze(0))[0, -1])

	before = [(c - u).abs().max().item() for k, (c, u) in enumerate(zip(cached, uncached))
		if prefix_len + k < maxlen]
	after = [(c - u).abs().max().item() for k, (c, u) in enumerate(zip(cached, uncached))
		if prefix_len + k >= maxlen]
	agree = sum(int(c.argmax()) == int(u.argmax()) for c, u in zip(cached, uncached))
	print(f'max_seq_len {maxlen}, prefix {prefix_len}: first trim at step {maxlen - prefix_len}')
	print(f'  before any trim: max |dlogit| {max(before):.3e}  ({len(before)} steps)')
	print(f'  after  trimming: max |dlogit| {max(after):.3e}  ({len(after)} steps)')
	print(f'  greedy argmax agreement {agree}/{len(cached)}'
		f' — {len(cached) - agree} step(s) emit a DIFFERENT token')
	print('EXPECTED DIVERGENCE (see KVDecoder._trim); the sliding translator never trims')


def packed_probe (device, seed):
	'''Guard the packed-sequence trap: attention_mask=None + no cache hides the source half.

	transformers' masking_utils._preprocess_mask_arguments runs find_packed_sequence_indices whenever
	`attention_mask is None and past_key_values is None`, and that helper calls ANY two consecutive
	position_ids not differing by exactly 1 a document boundary. `pos_style: absolute` jumps at <sep>
	(…-8054, -1, -1, 0…), so a maskless uncached forward blocks every target->source pair while
	TRAINING — which always passes masks (Seq2Seq2._collate_flat) — allowed them all.

	This is a silent, load-bearing difference, so it is asserted rather than merely measured:
	  ones-mask == cached prefill                (both are the training computation)
	  None-mask != either, under absolute        (the bug, if the fix is ever reverted)
	  all three agree under sep/flat             (monotone unit-step, no packing detected)
	'''
	from transformers.masking_utils import find_packed_sequence_indices

	torch.manual_seed(seed)
	model = MidiTranslator(vocab_size=838, d_model=128, n_layer=2, n_head=4, max_seq_len=4096,
		dropout=0.0).to(device).eval()
	n_src, n_tgt = 60, 12
	total = n_src + 1 + n_tgt
	ids = torch.randint(6, 838, (1, total), device=device)
	ones = torch.ones_like(ids)

	# the two layouts positions_for produces, at the shape build_prefix uses at step 0 (head=True)
	src = [p - 9000 - 1 for p in range(n_src - 1)]
	src = [src[0] - 1] + src
	absolute = src + [-1] + [-1] + list(range(n_tgt - 1))
	sep = list(range(-(n_src + 1), n_tgt))

	failures = []
	for label, pos in (('absolute', absolute), ('sep', sep)):
		tp = torch.tensor([pos], device=device)
		segments = int(find_packed_sequence_indices(tp)[0].max()) + 1
		with torch.no_grad():
			masked = model(ids, ones, tp)[0, -1, :]
			bare = model(ids, None, tp)[0, -1, :]
			cached = KVDecoder(model).prefill(ids[0].tolist(), pos, device=device)
		d_cache = (masked - cached).abs().max().item()
		d_bare = (masked - bare).abs().max().item()
		print(f'  {label:9s} segments {segments}  |ones-cached| {d_cache:.3e}  |ones-none| {d_bare:.3e}')
		if d_cache > 1e-4:
			failures.append(f'{label}: cached prefill disagrees with the training (ones-mask) path')
		if label == 'absolute' and d_bare < 1e-3:
			failures.append('absolute: maskless forward no longer differs — has positions_for changed, '
				'or did transformers drop the packed-sequence heuristic? Re-derive before trusting it.')
		if label == 'sep' and d_bare > 1e-4:
			failures.append('sep: maskless forward differs, so packing is now detected on a '
				'monotone layout')

	# and the mask itself: under absolute, no target position may reach any source position
	seg = find_packed_sequence_indices(torch.tensor([absolute]))[0]
	causal = torch.ones(total, total).tril().bool()
	allowed = causal & (seg.unsqueeze(0) == seg.unsqueeze(1))
	pairs = int(allowed[n_src + 1:, :n_src].sum())
	print(f'  absolute target->source pairs allowed with no mask: {pairs}'
		f' of {(total - n_src - 1) * n_src}')
	if pairs != 0:
		failures.append(f'absolute: expected 0 target->source pairs, got {pairs}')

	for f in failures:
		print(f'FAIL {f}')
	print('PACKED-MASK GUARD OK' if not failures else 'PACKED-MASK GUARD FAILED')
	return not failures


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--run', default=None, help='training run dir (omit for a random-init model)')
	ap.add_argument('--checkpoint', default=None)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--prefix', type=int, default=1200, help='prefix length (measured runs: ~1200)')
	ap.add_argument('--new', type=int, default=800, help='tokens to generate (measured runs: ~820)')
	ap.add_argument('--equiv-new', type=int, default=48,
		help='tokens for the per-step equivalence check (the uncached side is O(T^2), so keep it small)')
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--threads', type=int, default=0)
	ap.add_argument('--skip-timing', action='store_true')
	ap.add_argument('--trim-probe', action='store_true',
		help='measure the divergence across a cache trim instead of checking equivalence')
	ap.add_argument('--packed-probe', action='store_true',
		help='assert the packed-sequence mask trap stays fixed (see packed_probe)')
	args = ap.parse_args()

	if args.threads:
		torch.set_num_threads(args.threads)
	device = torch.device(args.device)

	if args.trim_probe:
		print('=== trim divergence probe ===')
		trim_probe(device, args.seed)
		return 0

	if args.packed_probe:
		print('=== packed-sequence mask guard ===')
		return 0 if packed_probe(device, args.seed) else 1

	model, name = build(args.run, args.checkpoint, device)
	vocab = model.vocab_size
	print(f'[model] {name}  vocab {vocab}  max_seq_len {model.max_seq_len}  device {device}'
		f'  threads {torch.get_num_threads()}')

	print(f'\n=== equivalence (prefix {args.prefix}, {args.equiv_new} steps) ===')
	diff, agree, total = check_equivalence(model, device, args.prefix, args.equiv_new, vocab, args.seed)
	print(f'max abs logit diff {diff:.3e}   argmax agreement {agree}/{total}')
	ok = agree == total and diff < 1e-3
	print('EQUIVALENT' if ok else 'DIVERGED')

	if not args.skip_timing:
		print(f'\n=== timing (prefix {args.prefix}, {args.new} new tokens) ===')
		t_on, out_on = time_route(model, device, args.prefix, args.new, vocab, args.seed, True)
		print(f'cached    {t_on:7.2f}s  {1000 * t_on / len(out_on):7.2f} ms/token  ({len(out_on)} tokens)')
		t_off, out_off = time_route(model, device, args.prefix, args.new, vocab, args.seed, False)
		print(f'uncached  {t_off:7.2f}s  {1000 * t_off / len(out_off):7.2f} ms/token  ({len(out_off)} tokens)')
		same = torch.equal(out_on, out_off)
		print(f'speedup   {t_off / t_on:7.2f}x   identical tokens: {same}')
		if not same:
			n = min(len(out_on), len(out_off))
			first = next((i for i in range(n) if int(out_on[i]) != int(out_off[i])), None)
			print(f'  first mismatch at {first} (lengths {len(out_on)} vs {len(out_off)})')
			ok = False

	return 0 if ok else 1


if __name__ == '__main__':
	sys.exit(main())
