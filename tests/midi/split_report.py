#!/usr/bin/env python3
'''Report what a config's `data.splits` string actually resolves to on the real corpus.

The filter is POSITIONAL — a sample lands in phase `index % cycle` — so the split depends on the
manifest's ORDER, not on any property of the samples. This reads the real name list and reports the
resulting sizes, so the config's stated counts can be checked rather than trusted.

Usage
    python3 tests/midi/split_report.py --config configs/<name>.local.yaml [--root /data/midi/nota1m]
'''

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

import yaml									# noqa: E402
from starry.utils.parsers import parseFilterStr	# noqa: E402


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', required=True)
	ap.add_argument('--root', default=None, help='corpus root; default = the config\'s data.root')
	ap.add_argument('--val-batch-size', type=int, default=None)
	args = ap.parse_args()

	config = yaml.safe_load(open(args.config))
	splits = config['data']['splits']
	root = args.root or config['data']['root']
	val_bs = args.val_batch_size or config['trainer']['val_batch_size']

	manifest = json.load(open(os.path.join(root, 'index.json')))

	# Replicate the feeder exactly (seq2seq2.py:538): names come from the manifest's per-shard lists,
	# then are globally SORTED and filtered to .txt. The sort is what makes the positional filter stable
	# — the manifest's shard iteration order does not reach the split.
	pooled = []
	for shard_names in manifest['shards'].values():
		pooled.extend(shard_names)
	names = sorted(set(pooled))
	names = [name for name in names if name.endswith('.txt')]
	n = len(names)

	print('config  : %s' % args.config)
	print('root    : %s' % root)
	print('splits  : %r' % splits)
	print('manifest: %d shards, samples=%s, arms %s'
		% (len(manifest['shards']), manifest.get('samples'), manifest.get('arms')))
	print('names   : %d pooled -> %d unique .txt (the list the split indexes into)' % (len(pooled), n))
	print()

	segments = splits.split(':')
	seen = {}
	for i, seg in enumerate(segments):
		shuffled = '*' in seg
		phases, cycle = parseFilterStr(seg)
		idx = [j for j in range(n) if j % cycle in phases]
		seen[i] = set(idx)
		role = 'rank %d (%s)' % (i, 'TRAIN' if i == 0 else 'VAL')
		print('%-16s %-14r shuffle=%-5s cycle=%-4d phases=%s'
			% (role, seg, shuffled, cycle, '%d..%d' % (phases[0], phases[-1]) if len(phases) > 1 else phases[0]))
		print('%-16s %d samples (%.2f%% of corpus)' % ('', len(idx), 100 * len(idx) / n))
		if i == 1 and val_bs:
			print('%-16s %d batches at val_batch_size %d' % ('', -(-len(idx) // val_bs), val_bs))
		print()

	# a sample in both segments would be trained on and validated against — silent leakage
	if len(seen) > 1:
		overlap = seen[0] & seen[1]
		print('overlap between train and val: %d %s' % (len(overlap), '(OK)' if not overlap else '<-- LEAK'))
		covered = len(seen[0] | seen[1])
		print('coverage: %d of %d samples (%d unused)' % (covered, n, n - covered))

	return 0


if __name__ == '__main__':
	sys.exit(main())
