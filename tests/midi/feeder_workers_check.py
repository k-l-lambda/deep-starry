#!/usr/bin/env python3
'''Does the seq2seq2 feeder survive `num_workers > 0`, and what would it buy?

`dataset_factory` builds every DataLoader with no `num_workers`, so parsing is serial with compute: at
bs 128 roughly 750-920 ms of a 2672 ms step is the GPU waiting on the main process. The handle cache is
already keyed by PID (seq2seq2.py:185) so forked workers would not share a file offset — the archive
side is prepared for this.

What is NOT prepared is device placement. `collateBatch` ends with `.to(self.device)`, and with
num_workers > 0 `collate_fn` runs IN THE WORKER, so a cuda device there means initialising CUDA inside a
forked process. This script measures rather than assumes:

  1. cuda collate + workers   -> expected to fail; this is the blocker to name precisely
  2. cpu collate + workers    -> the throughput actually available if placement moves out of collate
  3. cpu collate, no workers  -> baseline for (2), isolating the worker effect from the device effect

Run per-case so one crash does not hide the rest.

Usage
    DATA_DIR=/data/midi python3 tests/midi/feeder_workers_check.py --case cuda-workers --device cuda:0
'''

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from starry.utils.config import Configuration			# noqa: E402
from starry.midi.data.seq2seq2 import Seq2Seq2			# noqa: E402


ANY_SEQ2SEQ2_CONFIG = 'configs/midi-translator-seq2seq2.yaml'


def make_dataset (config, root, device):
	'''Build the TRAIN split only, at the device the case wants.'''
	args = config['data.args']
	train, _ = Seq2Seq2.load(root, args, splits=config['data.splits'],
		args_variant=config['data.args_variant'], device=device)
	return train


def timed (loader, batches, warmup=2):
	it = iter(loader)
	for _ in range(warmup):
		next(it)
	torch.cuda.synchronize() if torch.cuda.is_available() else None
	t0 = time.time()
	seen = 0
	lengths = []
	for _ in range(batches):
		batch = next(it)
		lengths.append(int(batch['input_ids'].shape[1]))
		seen += int(batch['input_ids'].shape[0])
	elapsed = time.time() - t0
	return elapsed, seen, lengths


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--case', required=True,
		choices=('cuda-workers', 'cpu-workers', 'cpu-serial'))
	ap.add_argument('--device', default='cuda:0')
	ap.add_argument('--batch-size', type=int, default=128)
	ap.add_argument('--workers', type=int, default=8)
	ap.add_argument('--batches', type=int, default=10)
	ap.add_argument('--data-dir', default=os.environ.get('DATA_DIR', '.'))
	ap.add_argument('--config', default=ANY_SEQ2SEQ2_CONFIG,
		help='any Seq2Seq2 config; only data.args/splits/root are read (default: %(default)s)')
	args = ap.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)
	root = os.path.join(args.data_dir, config['data.root'])

	# The point of the comparison: only the collate device and worker count differ.
	collate_device = args.device if args.case == 'cuda-workers' else 'cpu'
	workers = 0 if args.case == 'cpu-serial' else args.workers

	print('case            : %s' % args.case)
	print('collate device  : %s' % collate_device)
	print('num_workers     : %d' % workers)
	print('batch_size      : %d' % args.batch_size)

	dataset = make_dataset(config, root, collate_device)
	loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collateBatch,
		num_workers=workers)

	try:
		elapsed, seen, lengths = timed(loader, args.batches)
	except Exception as error:
		# The failure text is the deliverable for the cuda-workers case, so print it rather than raise.
		print('\nFAILED: %s: %s' % (type(error).__name__, str(error).split('\n')[0]))
		return 1

	per = elapsed / args.batches
	print('\nfeed-only: %d batches in %.2f s -> %.0f ms/batch, %.1f samples/s'
		% (args.batches, elapsed, per * 1000, seen / elapsed))
	print('T over these batches: median %d  max %d'
		% (sorted(lengths)[len(lengths) // 2], max(lengths)))
	return 0


if __name__ == '__main__':
	sys.exit(main())
