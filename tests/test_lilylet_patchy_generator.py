"""Smoke test: generate a small Lilylet file with LilyletPatchyGenerator.

Loads the trained NotaGen-X "large" checkpoint (lr0.2 run, fully trained at
epoch 999) and autoregressively decodes a short document, writing it to
tests/output/lilylet_generated.lyl.

Usage:
	python tests/test_lilylet_patchy_generator.py
	python tests/test_lilylet_patchy_generator.py --max-patches 96 --temperature 0.9
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from starry.lilylet.patchyGenerator import LilyletPatchyGenerator


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Checkpoint sync has stopped; this lr0.2 best.chkpt is currently stable and usable.
CKPT = '/home/camus/data/models/deep-starry-logs/lilylet/20260606-lilylet-notagenx-large-lr0.2/best.chkpt'
TOKENIZER = os.path.join(REPO_ROOT, 'assets', 'manual-tokenizer.json')

# NotaGen-X "large" hyperparameters (must match the checkpoint).
MODEL_ARGS = dict(
	char_vocab_size=256,
	patch_size=16,
	patch_length=1024,
	hidden_size=1280,
	patch_num_layers=20,
	char_num_layers=6,
	n_head=20,
)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('--checkpoint', default=CKPT)
	parser.add_argument('--tokenizer', default=TOKENIZER)
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

	print('checkpoint:', args.checkpoint)
	print('tokenizer: ', args.tokenizer)
	gen = LilyletPatchyGenerator.load(args.checkpoint, args.tokenizer, MODEL_ARGS)
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
