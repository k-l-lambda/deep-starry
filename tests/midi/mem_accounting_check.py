#!/usr/bin/env python3
'''Why does tune_probe's `reserved` disagree with nvidia-smi by ~3x?

ANSWERED: caching-allocator fragmentation under variable T. Keep this script for the per-term
breakdown, but the headline is settled and is recorded at trainerQuantitative.py's STARRY_MEM_TRACE
block, which reads the counters from inside the real loop:

    real trainer, bs 20     peak_alloc 11.15G   reserved 40.01G   ratio 3.59
    + expandable_segments   peak_alloc 11.14G   reserved 11.65G   ratio 1.045
    tune_probe (fixed T)    alloc      11.17G   reserved 11.56G   ratio 1.035

`peak_alloc` is the same everywhere: the memory REQUIREMENT never differed, so nothing was leaking
and no term below accounts for the gap. Only the pool differs. Real T varies per step, each unseen
shape carves fresh blocks, torch never returns a block to the driver, and the pool converges on the
union of shapes rather than the max. tune_probe hides this by measuring ONE fixed shape after
empty_cache() — hence its implausible 1.035, and hence the config memory tables being a lower bound
rather than a footprint. Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True collapses it.

This script separates the smaller terms, which turn out to be exactly that — small:

  A. CUDA context + libs      what nvidia-smi counts and torch's counters do not   (+0.83G measured)
  B. shape variety            pool growth, as above, but bounded here because the fixed-shape pass
                              runs FIRST in the same process and pre-carves large blocks the real
                              batches then reuse — the reason this script saw only +0.84G where the
                              real trainer, which never sees a padded 2048 batch, reaches +29G
  C. the real feeder's T      the probe pads to the 2048 cap, real batches sit lower (median ~1800)

Point B is the trap worth remembering: measuring a fixed worst-case shape before the varying ones
PRIMES the allocator and understates fragmentation. Order of measurement changed the result.

Reports torch's own numbers and the process RSS-on-device side by side, for fixed-shape and then
real-feeder steps in the SAME process, so the deltas are attributable.

Usage
    DATA_DIR=/data/midi python3 tests/midi/mem_accounting_check.py \
        --config configs/midi-translator-seq2seq2.yaml --device cuda:0
'''

import argparse
import os
import subprocess
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from starry.utils.config import Configuration				# noqa: E402
from starry.utils.dataset_factory import loadDataset		# noqa: E402
from starry.utils.model_factory import loadModel			# noqa: E402


def smi_self (device_index):
	'''What nvidia-smi attributes to THIS pid — the number that disagrees with torch's counters.'''
	try:
		out = subprocess.check_output(['nvidia-smi',
			'--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
			stderr=subprocess.DEVNULL).decode()
	except Exception:
		return None
	me = str(os.getpid())
	for line in out.strip().split('\n'):
		parts = [p.strip() for p in line.split(',')]
		if len(parts) >= 2 and parts[0] == me:
			return float(parts[1].split()[0]) / 1024		# MiB -> GiB
	return None


def report (tag, device):
	alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 3
	res = torch.cuda.max_memory_reserved(device) / 1024 ** 3
	cur_res = torch.cuda.memory_reserved(device) / 1024 ** 3
	smi = smi_self(device.index or 0)
	print('  %-26s alloc %6.2fG  reserved %6.2fG (now %6.2fG)  ratio %.3f  nvidia-smi %s'
		% (tag, alloc, res, cur_res, res / alloc if alloc else 0,
			'%6.2fG' % smi if smi is not None else 'n/a'))
	return alloc, res, smi


def one_step (model, opt, batch, grad_clip=False, metrics=False):
	opt.zero_grad()
	loss, metric = model(batch)
	loss.backward()
	if grad_clip:
		# The trainer ALWAYS calls this, passing inf when trainer.grad_clip is unset, because the
		# pre-clip norm is wanted as a metric. It materializes a norm over every gradient.
		gn = torch.nn.utils.clip_grad_norm_(
			[p for p in model.parameters() if p.requires_grad], float('inf'))
	opt.step()
	if metrics:
		m = metric if type(metric) == dict else {'acc': metric}
		if grad_clip:
			m = {**m, 'grad_norm': float(gn)}
		return float(loss.item()), m
	return float(loss.item()), None


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', required=True)
	ap.add_argument('--device', default='cuda:0')
	ap.add_argument('--steps', type=int, default=40)
	ap.add_argument('--data-dir', default=os.environ.get('DATA_DIR', '.'))
	# Bisection against the real trainer path (trainerQuantitative.py:220-247). The probe's own loop
	# reproduces 13.2G while the real trainer holds 28.0G on the same config, so the difference is one of
	# these elements, not the model or the batch size.
	ap.add_argument('--real-optim', action='store_true',
		help="use optim() from config['optim'] instead of a plain AdamW")
	ap.add_argument('--grad-clip', action='store_true',
		help='call clip_grad_norm_(max_norm=inf) each step, as the trainer does for its norm metric')
	ap.add_argument('--metrics', action='store_true',
		help='consume the metric dict the loss returns, as the trainer does')
	args = ap.parse_args()

	device = torch.device(args.device)
	config = Configuration.createOrLoad(args.config, volatile=True)
	bs = config['data.batch_size']
	cap = int((config['data.args'] or {}).get('max_tokens') or 2048)

	train, _ = loadDataset(config, data_dir=args.data_dir, device=device, batch_size=bs)
	model = loadModel(config['model'], postfix='Loss').to(device)
	if args.real_optim:
		from starry.utils.optim import optim				# noqa: E402
		opt = optim(config['optim'], model.parameters(), init_step=0)
	else:
		opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

	print('config    : %s' % args.config)
	print('bs %d  cap T %d  device %s' % (bs, cap, torch.cuda.get_device_name(device)))
	print('flags     : real_optim=%s grad_clip=%s metrics=%s'
		% (args.real_optim, args.grad_clip, args.metrics))
	print('\nBaseline: model + optimizer state resident, no step taken yet')
	report('after .to(device)', device)

	# --- A: the fixed-shape worst case, exactly what tune_probe section 1 measures.
	it = iter(train)
	base = next(it)
	pad_id = getattr(model, 'pad_id', 0)
	fixed = {}
	for k, v in base.items():
		if torch.is_tensor(v):
			v = v[:1].repeat(bs, *([1] * (v.dim() - 1)))
			if v.dim() >= 2 and v.shape[1] < cap:
				pad = torch.full((v.shape[0], cap - v.shape[1], *v.shape[2:]), pad_id,
					dtype=v.dtype, device=v.device)
				v = torch.cat([v, pad], dim=1)
			fixed[k] = v
		else:
			fixed[k] = v

	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats(device)
	for _ in range(5):
		one_step(model, opt, fixed, args.grad_clip, args.metrics)
	torch.cuda.synchronize(device)
	print('\nA. FIXED shape T=%d, repeated row (tune_probe section 1)' % cap)
	a_alloc, a_res, a_smi = report('5 identical steps', device)

	# --- B/C: real feeder, variable T. NO empty_cache and NO reset: the point is to watch the pool grow.
	print('\nB. REAL feeder, variable T — pool growth across %d steps (no empty_cache)' % args.steps)
	torch.cuda.reset_peak_memory_stats(device)
	seen = []
	for i in range(args.steps):
		try:
			batch = next(it)
		except StopIteration:
			it = iter(train)
			batch = next(it)
		one_step(model, opt, batch, args.grad_clip, args.metrics)
		seen.append(int(batch['input_ids'].shape[1]))
		if (i + 1) % 10 == 0:
			torch.cuda.synchronize(device)
			report('after %2d real steps' % (i + 1), device)

	print('\nT over real steps: median %d  min %d  max %d  distinct %d'
		% (sorted(seen)[len(seen) // 2], min(seen), max(seen), len(set(seen))))

	b_alloc, b_res, b_smi = report('final', device)

	print('\n--- accounting')
	print('fixed-shape reserved      %6.2fG   <- what the config table records' % a_res)
	print('real-feeder reserved      %6.2fG   delta %+.2fG from shape variety' % (b_res, b_res - a_res))
	if b_smi is not None:
		print('nvidia-smi for this pid   %6.2fG   delta %+.2fG over torch reserved (context + libs)'
			% (b_smi, b_smi - b_res))
	print('\nDistinct T values drive the allocator pool: a fixed shape reserves one block set, a varying')
	print('one reserves toward the union. That is the term the config table cannot predict.')


if __name__ == '__main__':
	sys.exit(main())
