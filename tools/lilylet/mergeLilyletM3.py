'''Merge per-GPU shards produced by preprocessLilyletM3.py (--start/--end slicing)
into a single lilylet-m3-abc-pooled artifact, so LilyletM3Distill can load one root.

Each shard is a dict with the same top-level layout (version/format/tokenizer/
config/items/stats). We concatenate `items` in shard order, keep the first shard's
tokenizer/config (asserting they agree), and recompute stats. The per-shard `slice`
under config is dropped; a `shards` list records provenance.

Usage (run from repo root):
  python3 -m tools.lilylet.mergeLilyletM3 OUT.pt SHARD1.pt SHARD2.pt ...
  python3 -m tools.lilylet.mergeLilyletM3 OUT.pt --glob "DIR/notagen-100k.shard-g*.pt"
'''

import argparse
import glob as globlib
import logging
import os
import sys

import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('output_path', type=str, help='merged artifact path (.pt)')
	parser.add_argument('shards', type=str, nargs='*', help='shard .pt files in desired order')
	parser.add_argument('--glob', type=str, default=None, help='glob pattern for shards (sorted); overrides positional shards')
	args = parser.parse_args()

	paths = sorted(globlib.glob(args.glob)) if args.glob else args.shards
	if not paths:
		raise SystemExit('no shard files given')
	logging.info('Merging %d shards:', len(paths))
	for p in paths:
		logging.info('  %s', p)

	base = None
	items = []
	total_files = 0
	missing_abc = 0
	unknown_total = 0
	shard_meta = []
	for p in paths:
		shard = torch.load(p, map_location='cpu', weights_only=False)
		if base is None:
			base = shard
		else:
			# sanity: tokenizer + core m3 config must agree across shards
			assert shard['tokenizer'] == base['tokenizer'], f'tokenizer mismatch in {p}'
			assert shard['config']['patch_size'] == base['config']['patch_size'], f'patch_size mismatch in {p}'
		n = len(shard['items'])
		items.extend(shard['items'])
		st = shard.get('stats', {})
		total_files += st.get('files', n)
		missing_abc += st.get('missing_abc', 0)
		unknown_total += st.get('unknown_total', 0)
		shard_meta.append(dict(path=os.path.basename(p), items=n, slice=shard['config'].get('slice')))
		logging.info('  + %s: %d items', os.path.basename(p), n)

	config = dict(base['config'])
	config.pop('slice', None)
	config['shards'] = shard_meta

	artifact = dict(
		version=base['version'],
		format=base['format'],
		tokenizer=base['tokenizer'],
		config=config,
		items=items,
		stats=dict(files=total_files, paired=len(items), missing_abc=missing_abc, unknown_total=unknown_total),
	)
	os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
	tmp = args.output_path + '.tmp'
	torch.save(artifact, tmp)
	os.replace(tmp, args.output_path)
	logging.info('Wrote %s: %d items (missing_abc=%d, unknown_total=%d)',
		args.output_path, len(items), missing_abc, unknown_total)


if __name__ == '__main__':
	main()
