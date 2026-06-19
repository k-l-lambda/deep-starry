'''Preprocess paired Lilylet + ABC into a single artifact for cross-modal use.

For every `.lyl` file under `lilylet_dir`, this pairs it (by mirrored relative
path) with the matching `.abc` file under `abc_dir` and emits, per sample:
  - `patches`: Lilylet tokenize + patchize at the M3 patch size (default 64),
    via starry.lilylet.data.patchifier.patchify_text (uint8 [P, patch_size]).
  - `m3_embedding`: the mean-pooled CLaMP 3 M3 symbolic embedding of the ABC
    file (float16 [768]) — `encode_abc`'s un-pooled per-patch output averaged
    over real patches (masked mean, matching CLaMP's avg_pooling). `m3_patches`
    records the original (pre-pooling) patch count.

Pairing: `lilylet_dir` and `abc_dir` are assumed to MIRROR each other — a
.lyl at `<lilylet_dir>/A/B/hash.lyl` pairs with `<abc_dir>/A/B/hash.abc`. If the
mirrored path is absent we fall back to a flat `<abc_dir>/hash.abc`.

Output is a single `.pt` artifact (version 1), mirroring the single-file layout of
tools/lilylet/preprocessLilylet.py's pack_lilylet_notagen. For multi-GPU runs,
slice the (sorted, deterministic) file list with --start/--end and give each
process its own --device and output path; merge the shards afterwards.

Run from the repo root as a module, e.g.:
  python3 -m tools.lilylet.preprocessLilyletM3 configs/lilylet-notagen-data.yaml \\
      /data1/datasets/nota/lilylet/lyl \\
      /data1/datasets/nota/NotaGenX-opus/abc \\
      output/lilylet-m3-abc.shard0.pt --start 0 --end 25000 --device cuda:2
'''

import argparse
import logging
import os
import sys

import torch
from tqdm import tqdm

from starry.lilylet.data.patchifier import LilyletTokenizer, patchify_text, find_lilylet_files
from starry.lilylet.m3 import load_m3_encoder, M3Patchilizer, encode_abc, M3_HIDDEN_SIZE, PATCH_SIZE as M3_PATCH_SIZE
from starry.utils.config import Configuration


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def _resolve_abc (lyl_path, lilylet_dir, abc_dir):
	'''Map a .lyl path to its paired .abc path by mirroring the relative directory
	structure (lilylet_dir and abc_dir share the same A/B/hash bucketing). Falls
	back to a flat <abc_dir>/hash.abc if the mirrored path is absent. Returns
	(rel, abc_path) where rel is the .lyl path relative to lilylet_dir.'''
	rel = os.path.relpath(lyl_path, lilylet_dir)
	mirrored = os.path.join(abc_dir, os.path.splitext(rel)[0] + '.abc')
	if os.path.exists(mirrored):
		return rel, mirrored
	base = os.path.splitext(os.path.basename(lyl_path))[0]
	flat = os.path.join(abc_dir, base + '.abc')
	return rel, flat


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str, help='config of Lilylet data (for tokenizer_path / patch_length / patch_stream)')
	parser.add_argument('lilylet_dir', type=str, help='directory containing .lyl files (recursed)')
	parser.add_argument('abc_dir', type=str, help='directory containing matching .abc files (mirrored relative path)')
	parser.add_argument('output_path', type=str, help='output artifact path (.pt)')
	parser.add_argument('--limit', type=int, default=0, help='process at most N paired samples after slicing (0 = all)')
	parser.add_argument('--start', type=int, default=0, help='start index into the sorted .lyl file list (inclusive)')
	parser.add_argument('--end', type=int, default=0, help='end index into the sorted .lyl file list (exclusive; 0 = to end)')
	parser.add_argument('--flush-every', type=int, default=2000, help='checkpoint the output every N processed items (0 = only at end)')
	parser.add_argument('--patch-size', type=int, default=M3_PATCH_SIZE, help='Lilylet patch size (default = M3 patch size %d)' % M3_PATCH_SIZE)
	parser.add_argument('--m3-weights', type=str, default=None, help='path to CLaMP 3 weights (default: auto-glob cached saas)')
	parser.add_argument('--device', type=str, default=None, help='torch device for the M3 encoder (default: cuda if available)')
	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)
	tokenizer_path = config['data.args']['tokenizer_path']
	patch_length = config['data.args'].get('patch_length', 2048)
	patch_stream = config['data.args'].get('patch_stream', True)
	patch_size = args.patch_size

	tokenizer = LilyletTokenizer(tokenizer_path)
	vocab_size = max(entry['id'] for entry in tokenizer.vocab) + 1

	logging.info('Indexing .lyl files under %s ...', args.lilylet_dir)
	lyl_files = find_lilylet_files(args.lilylet_dir)
	total_found = len(lyl_files)

	start = max(0, args.start)
	end = args.end if args.end and args.end > 0 else len(lyl_files)
	lyl_files = lyl_files[start:end]
	if args.limit and args.limit > 0:
		lyl_files = lyl_files[:args.limit]
	logging.info('Found %d .lyl files; processing slice [%d:%d] -> %d files',
		total_found, start, end, len(lyl_files))

	logging.info('Loading M3 encoder...')
	encoder = load_m3_encoder(weights_path=args.m3_weights, device=args.device)
	device = next(encoder.parameters()).device
	patchilizer = M3Patchilizer(syntax='abc')
	logging.info('M3 encoder ready on %s (weights=%s)', device, os.path.basename(encoder._m3_weights_path))

	os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

	def save (items, missing_abc, unknown_total):
		artifact = dict(
			version=1,
			format='lilylet-m3-abc-pooled',
			tokenizer=dict(path=tokenizer_path, vocab_size=vocab_size),
			config=dict(
				patch_size=patch_size,
				patch_length=patch_length,
				patch_stream=patch_stream,
				slice=dict(start=start, end=end),
				m3=dict(weights=encoder._m3_weights_path, hidden=M3_HIDDEN_SIZE, patch_size=M3_PATCH_SIZE, pooling='mean'),
			),
			items=items,
			stats=dict(files=len(lyl_files), paired=len(items), missing_abc=missing_abc, unknown_total=unknown_total),
		)
		tmp = args.output_path + '.tmp'
		torch.save(artifact, tmp)
		os.replace(tmp, args.output_path)

	items = []
	unknown_total = 0
	missing_abc = 0
	processed = 0
	for lyl_path in tqdm(lyl_files):
		rel, abc_path = _resolve_abc(lyl_path, args.lilylet_dir, args.abc_dir)
		if not os.path.exists(abc_path):
			logging.warning('no ABC pair for %s (expected %s)', rel, abc_path)
			missing_abc += 1
			continue

		with open(lyl_path, 'r', encoding='utf-8') as f:
			lyl_text = f.read()
		# Drop the leading `%<style>` prompt lines (period/composer/instrumentation):
		# these Lilylet patches are paired with the ABC M3 embedding, which encodes
		# real musical content (ABC's M3 preprocessing strips `%` comments too). The
		# style prompt is user-facing conditioning, not music, so excluding it keeps
		# both sides of the pair semantically aligned.
		try:
			patches, unknowns = patchify_text(
				lyl_text, tokenizer, file=rel,
				patch_size=patch_size, patch_length=patch_length, patch_stream=patch_stream,
				drop_style_comments=True,
			)
			# encode_abc returns un-pooled [num_patches, 768] with padding already
			# stripped, so mean over dim 0 is the masked average pooling (matches
			# CLaMP's avg_pooling over real patches). Store the [768] global vector.
			m3 = encode_abc(abc_path, encoder, patchilizer, device=device)
		except Exception as e:
			logging.warning('failed on %s: %s', rel, e)
			continue
		m3_patches = int(m3.shape[0])
		m3 = m3.mean(dim=0).to('cpu', torch.float16)

		items.append(dict(
			path=rel,
			patches=patches,
			m3_embedding=m3,
			m3_patches=m3_patches,
			unknowns=unknowns,
		))
		unknown_total += sum(hit['count'] for hit in unknowns)
		processed += 1
		if args.flush_every and processed % args.flush_every == 0:
			save(items, missing_abc, unknown_total)
			logging.info('checkpoint: %d items saved to %s', len(items), args.output_path)

	save(items, missing_abc, unknown_total)

	logging.info('Wrote %s', args.output_path)
	logging.info('Paired: %d / %d  (missing ABC: %d)', len(items), len(lyl_files), missing_abc)
	logging.info('Unknown total: %d', unknown_total)


if __name__ == '__main__':
	main()
