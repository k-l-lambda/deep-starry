"""Smoke test: generate a small Lilylet file with LilyletPatchyGenerator.

Loads a trained NotaGen-X checkpoint and autoregressively decodes a short
document, writing it to tests/output/lilylet_generated.lyl.

The model architecture (base_type, hidden_size, heads, etc.) is read from the
checkpoint run's own `.state.yaml` — `Configuration.createOrLoad(<run dir>)`
loads it — so the script never hardcodes hyperparameters and works for any
backbone (gpt2 / llama / GQA) without extra flags.

Usage:
	python tests/test_lilylet_patchy_generator.py
	python tests/test_lilylet_patchy_generator.py --checkpoint <run>/best.chkpt --measures 8 --postprocess
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from starry.utils.config import Configuration
from starry.lilylet.patchyGenerator import LilyletPatchyGenerator


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Checkpoint sync has stopped; this lr0.2 best.chkpt is currently stable and usable.
CKPT = '/home/camus/data/models/deep-starry-logs/lilylet/20260606-lilylet-notagenx-large-lr0.2/best.chkpt'


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default=CKPT)
	parser.add_argument('--config', default=None,
		help="config to read model.args from; default = the checkpoint run's own "
		     "directory (its .state.yaml), so the architecture always matches")
	parser.add_argument('--tokenizer', default=None,
		help='override tokenizer path; default = the one recorded in the config')
	parser.add_argument('--max-patches', type=int, default=1024,
		help='patch cap; default 1024 = model patch_length, so the model stops on its own EOS')
	parser.add_argument('--temperature', type=float, default=0.9)
	parser.add_argument('--top-k', type=int, default=20)
	parser.add_argument('--top-p', type=float, default=0.95)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--prompt', default='', help='optional metadata header lines')
	parser.add_argument('--measures', type=int, default=None,
		help='force the body to start at [r:0/<measures>; default lets the model decide')
	parser.add_argument('--postprocess', action='store_true',
		help='drop [r:x/y] markers and insert blank lines after meta / at measure boundaries')
	parser.add_argument('--out', default=os.path.join(REPO_ROOT, 'tests', 'output', 'lilylet_generated.lyl'))
	args = parser.parse_args()

	torch.manual_seed(args.seed)

	# Read the run's recorded state: the checkpoint's directory holds a .state.yaml
	# with the full model.args (base_type, heads, intermediate_size, ...). Passing a
	# non-.yaml path to createOrLoad loads that, so the architecture always matches
	# the weights and we never hardcode hyperparameters here.
	config_src = args.config or os.path.dirname(args.checkpoint)
	config = Configuration.createOrLoad(config_src, volatile=True)

	# The config stores a repo-relative tokenizer path; resolve it absolutely so it
	# doesn't depend on the cwd (LilyletPatchyGenerator also hardens this).
	tokenizer_path = args.tokenizer
	if tokenizer_path is None:
		tk = config['data.args.tokenizer_path']
		tokenizer_path = tk if os.path.isabs(tk) else os.path.join(REPO_ROOT, tk)

	print('checkpoint:', args.checkpoint)
	print('config src:', config_src)
	print('base_type: ', config['model.args.base_type'] or 'gpt2')
	print('tokenizer: ', tokenizer_path)
	gen = LilyletPatchyGenerator.from_config(config, args.checkpoint, tokenizer_path=tokenizer_path)
	print('device:', gen.device, '| patch_size:', gen.patch_size,
		'| pad/bos/eos:', gen.pad_id, gen.bos_id, gen.eos_id)

	print(f'\n--- generating (max_patches={args.max_patches}, temp={args.temperature}, '
		f'top_k={args.top_k}, top_p={args.top_p}, measures={args.measures}, '
		f'postprocess={args.postprocess}) ---\n')
	text = gen.generate(
		prompt_text=args.prompt,
		max_patches=args.max_patches,
		temperature=args.temperature,
		top_k=args.top_k,
		top_p=args.top_p,
		measures=args.measures,
		postprocess=args.postprocess,
		verbose=True,
	)

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	with open(args.out, 'w', encoding='utf-8') as f:
		f.write(text)

	n_lines = text.count('\n') + 1
	print(f'\n\n===== generated {len(text)} chars, {n_lines} lines -> {args.out} =====')
	assert len(text) > 0, 'generation produced empty output'
	print('OK')


if __name__ == '__main__':
	main()
