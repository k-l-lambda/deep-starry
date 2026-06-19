'''Preprocess paired Lilylet + ABC into a single artifact for cross-modal use.

For every `.lyl` file under `lilylet_dir`, this pairs it (by basename) with the
matching `.abc` file under `abc_dir` and emits, per sample:
  - `patches`: Lilylet tokenize + patchize at the M3 patch size (default 64),
    via starry.lilylet.data.patchifier.patchify_text (uint8 [P, patch_size]).
  - `m3_embedding`: the mean-pooled CLaMP 3 M3 symbolic embedding of the ABC
    file (float16 [768]) — `encode_abc`'s un-pooled per-patch output averaged
    over real patches (masked mean, matching CLaMP's avg_pooling). `m3_patches`
    records the original (pre-pooling) patch count.

Output is a single `.pt` artifact (version 1), mirroring the single-file layout of
tools/lilylet/preprocessLilylet.py's pack_lilylet_notagen.

Run from the repo root as a module, e.g.:
  python3 -m tools.lilylet.preprocessLilyletM3 configs/lilylet-notagen-data.yaml \\
      /home/camus/work/lilylet/tests/output/notagenx-from-abc \\
      /home/camus/data/abc/notagenx-samples \\
      output/lilylet-m3-abc.pt
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


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str, help='config of Lilylet data (for tokenizer_path / patch_length / patch_stream)')
	parser.add_argument('lilylet_dir', type=str, help='directory containing .lyl files')
	parser.add_argument('abc_dir', type=str, help='directory containing matching .abc files (paired by basename)')
	parser.add_argument('output_path', type=str, help='output artifact path (.pt)')
	parser.add_argument('--limit', type=int, default=0, help='process at most N paired samples (0 = all)')
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

	logging.info('Loading M3 encoder...')
	encoder = load_m3_encoder(weights_path=args.m3_weights, device=args.device)
	device = next(encoder.parameters()).device
	patchilizer = M3Patchilizer(syntax='abc')
	logging.info('M3 encoder ready on %s (weights=%s)', device, os.path.basename(encoder._m3_weights_path))

	lyl_files = find_lilylet_files(args.lilylet_dir)
	if args.limit and args.limit > 0:
		lyl_files = lyl_files[:args.limit]
	logging.info('Found %d .lyl files under %s', len(lyl_files), args.lilylet_dir)

	items = []
	unknown_total = 0
	missing_abc = 0
	for lyl_path in tqdm(lyl_files):
		rel = os.path.relpath(lyl_path, args.lilylet_dir)
		base = os.path.splitext(os.path.basename(lyl_path))[0]
		abc_path = os.path.join(args.abc_dir, base + '.abc')
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
		patches, unknowns = patchify_text(
			lyl_text, tokenizer, file=rel,
			patch_size=patch_size, patch_length=patch_length, patch_stream=patch_stream,
			drop_style_comments=True,
		)

		# encode_abc returns un-pooled [num_patches, 768] with padding already
		# stripped, so mean over dim 0 is the masked average pooling (matches
		# CLaMP's avg_pooling over real patches). Store the [768] global vector.
		m3 = encode_abc(abc_path, encoder, patchilizer, device=device)
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

	os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
	artifact = dict(
		version=1,
		format='lilylet-m3-abc-pooled',
		tokenizer=dict(path=tokenizer_path, vocab_size=vocab_size),
		config=dict(
			patch_size=patch_size,
			patch_length=patch_length,
			patch_stream=patch_stream,
			m3=dict(weights=encoder._m3_weights_path, hidden=M3_HIDDEN_SIZE, patch_size=M3_PATCH_SIZE, pooling='mean'),
		),
		items=items,
		stats=dict(files=len(lyl_files), paired=len(items), missing_abc=missing_abc, unknown_total=unknown_total),
	)
	torch.save(artifact, args.output_path)

	logging.info('Wrote %s', args.output_path)
	logging.info('Paired: %d / %d  (missing ABC: %d)', len(items), len(lyl_files), missing_abc)
	logging.info('Unknown total: %d', unknown_total)


if __name__ == '__main__':
	main()
