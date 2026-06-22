'''Generate one Lilylet score per (model, prompt group) at the SAME first-sample seed
used by scoreLilyNotaByPrompt.py's final mode, and save each as a .lyl file.

For the final run, group gi's i-th sample used seed = seed_base + gi*10000 + (i+1)
(the loop does seed += 1 before the first generate), so the FIRST sample of group gi
is seed = seed_base + gi*10000 + 1. With seed_base=1000 that is 1001, 10001, 20001,
30001, 40001 for the top-5 groups. We reproduce exactly that first draw here.

Generation backend = ORT int8 KV (the deployment path), identical to the scored run.

Output: ~/data/lilylet/<out-subdir>/<model>/<NN>_<period>_<composer>_<instrument>.lyl
(plus a manifest.json mapping each file to its model/group/seed).

Run from repo root in the venv:
  python3 -m tools.lilylet.genSamplesByPrompt --out-subdir lilynota-samples-T1.5
'''

import argparse
import json
import logging
import os
import sys

import torch

from starry.utils.config import Configuration
from starry.lilylet.patchyGenerator import LilyletPatchyGenerator


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the three models: (label, run/onnx dir holding .state.yaml + best.chkpt + onnx/)
MODELS = [
	('lilynota', '/home/camus/data/models/LilyNota'),
	('lr0.2', '/home/camus/data/models/deep-starry-logs/lilylet/20260617-lilylet-notagenx-1m0617-llama-l4+10-lr0.2'),
	('d512', '/home/camus/data/models/deep-starry-logs/lilylet/20260617-lilylet-notagenx-1m0617-llama-l4+10-d512'),
]


def _seed_text (period, composer, instrumentation):
	return '%%%s\n%%%s\n%%%s' % (period, composer, instrumentation)


def _load_ort (run_dir, threads):
	sys.path.insert(0, os.path.join(REPO, 'tests', 'lilylet'))
	from bench_lilylet_int8_ort import ORTGeneratorKV
	config = Configuration.createOrLoad(run_dir, volatile=True)
	tok_path = config['data.args.tokenizer_path']
	if not os.path.isabs(tok_path):
		tok_path = os.path.join(REPO, tok_path)
	ckpt = os.path.join(run_dir, 'best.chkpt')
	onnx_dir = os.path.join(run_dir, 'onnx')
	torch.set_num_threads(threads)
	tgen = LilyletPatchyGenerator.from_config(config, ckpt, tokenizer_path=tok_path, device='cpu')
	patch_kv = os.path.join(onnx_dir, 'patch_kv_int8.onnx')
	token_kv = os.path.join(onnx_dir, 'token_kv_int8.onnx')
	token_full = os.path.join(onnx_dir, 'token_int8.onnx')
	token_fallback = token_full if os.path.isfile(token_full) else token_kv
	return ORTGeneratorKV(tgen, patch_kv, token_fallback, threads=threads, token_kv_onnx=token_kv)


def _slug (s):
	return ''.join(c if c.isalnum() else '-' for c in s).strip('-')


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--ref', default='/home/camus/data/lilylet/m3/notagen-100k.abc-vs-lyl.pt',
		help='prompt-avg artifact (for the top-K group ordering, same as the scored run)')
	ap.add_argument('--out-subdir', default='lilynota-samples-T1.5', help='new subdir under ~/data/lilylet')
	ap.add_argument('--top', type=int, default=5)
	ap.add_argument('--seed-base', type=int, default=1000)
	ap.add_argument('--temperature', type=float, default=1.5)
	ap.add_argument('--top-k', type=int, default=20)
	ap.add_argument('--top-p', type=float, default=0.95)
	ap.add_argument('--max-patches', type=int, default=1024)
	ap.add_argument('--measures', type=int, default=None)
	ap.add_argument('--threads', type=int, default=14)
	ap.add_argument('--postprocess', action='store_true', default=True, help='clean [r:x/y] markers + blank lines for readability')
	ap.add_argument('--raw', dest='postprocess', action='store_false', help='keep raw [r:x/y]-marked text')
	args = ap.parse_args()

	out_root = os.path.join(os.path.expanduser('~/data/lilylet'), args.out_subdir)
	os.makedirs(out_root, exist_ok=True)

	ref = torch.load(args.ref, map_location='cpu', weights_only=False)
	ordered = sorted(ref['prompts'], key=lambda p: p['count'], reverse=True)[:args.top]
	logging.info('Top-%d groups: %s', args.top, [p['composer'] for p in ordered])

	manifest = []
	for label, run_dir in MODELS:
		logging.info('=== model %s (%s) ===', label, run_dir)
		gen = _load_ort(run_dir, args.threads)
		mdir = os.path.join(out_root, label)
		os.makedirs(mdir, exist_ok=True)
		for gi, g in enumerate(ordered):
			seed = args.seed_base + gi * 10000 + 1     # first-sample seed of group gi
			seed_txt = _seed_text(g['period'], g['composer'], g['instrumentation'])
			torch.manual_seed(seed)
			text = gen.generate(prompt_text=seed_txt, max_patches=args.max_patches,
				temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
				measures=args.measures, postprocess=args.postprocess)
			fname = '%02d_%s_%s_%s.lyl' % (gi, _slug(g['period']), _slug(g['composer']), _slug(g['instrumentation']))
			fpath = os.path.join(mdir, fname)
			with open(fpath, 'w', encoding='utf-8') as f:
				f.write(text)
			n_lines = text.count('\n') + 1
			logging.info('  [%d] seed=%d -> %s (%d chars, %d lines)', gi, seed, fname, len(text), n_lines)
			manifest.append(dict(model=label, group_index=gi, seed=seed,
				period=g['period'], composer=g['composer'], instrumentation=g['instrumentation'],
				file=os.path.relpath(fpath, out_root), chars=len(text), lines=n_lines))

	with open(os.path.join(out_root, 'manifest.json'), 'w', encoding='utf-8') as f:
		json.dump(dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
			max_patches=args.max_patches, postprocess=args.postprocess, backend='ort-int8-kv',
			samples=manifest), f, indent=2, ensure_ascii=False)
	logging.info('Wrote %d files + manifest.json to %s', len(manifest), out_root)


if __name__ == '__main__':
	main()
