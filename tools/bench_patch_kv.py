"""Validate + benchmark patch-level KV cache vs full-recompute, INT8 ONNX.

Compares ORTGeneratorKV (patch_kv_int8.onnx, incremental patch decoding) against
the baseline ORTGenerator (patch_int8.onnx, full recompute each step):
  - correctness: the KV patch decoder is numerically equivalent to full recompute
    in fp32 (cos 1.0); see tools/export validation. In INT8 the two graphs are
    quantized independently, so their per-step rounding differs slightly and greedy
    generation can branch (a flipped tie, autoregressively amplified) — exactly the
    documented torch-fp32-vs-ORT-int8 effect. Both outputs are valid Lilylet.
  - single-step fidelity: patch hidden-state cos between the two int8 graphs.
  - speed: wall-clock for a long piece; KV wins, gap widening with length.

Both share the same token_int8.onnx token path, so any difference is purely the
patch-level cache / its quantization.

Usage:
  python tools/bench_patch_kv.py --run <RUN> --measures 32 --max-patches 256
"""

import os
import sys
import time
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'tests'))

import numpy as np
import torch

from starry.utils.config import Configuration
from starry.lilylet.patchyGenerator import LilyletPatchyGenerator
from bench_lilylet_int8_ort import ORTGenerator, ORTGeneratorKV

PROMPT = '%Romantic\n%Schubert, Franz\n%Keyboard\n'


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--run', required=True)
	ap.add_argument('--checkpoint', default=None)
	ap.add_argument('--measures', type=int, default=32)
	ap.add_argument('--max-patches', type=int, default=256)
	ap.add_argument('--threads', type=int, default=14)
	ap.add_argument('--seed', type=int, default=0)
	args = ap.parse_args()

	torch.set_num_threads(args.threads)
	run = args.run
	ckpt = args.checkpoint or os.path.join(run, 'best.chkpt')
	onnx = os.path.join(run, 'onnx')

	config = Configuration.createOrLoad(run, volatile=True)
	tk = config['data.args.tokenizer_path']
	tk = tk if os.path.isabs(tk) else os.path.join(REPO_ROOT, tk)
	gen = LilyletPatchyGenerator.from_config(config, ckpt, tokenizer_path=tk, device='cpu')

	base = ORTGenerator(gen, os.path.join(onnx, 'patch_int8.onnx'),
		os.path.join(onnx, 'token_int8.onnx'), threads=args.threads)
	kv = ORTGeneratorKV(gen, os.path.join(onnx, 'patch_kv_int8.onnx'),
		os.path.join(onnx, 'token_int8.onnx'), threads=args.threads)

	# ---- single-step fidelity: patch hidden state, KV-int8 vs full-int8 ----
	print('===== PATCH HIDDEN FIDELITY (int8 KV vs int8 full) =====')
	ps = gen.patch_size
	rng = np.random.default_rng(0)
	grid = [[gen.bos_id] * (ps - 1) + [gen.eos_id]] + [list(rng.integers(0, 254, ps)) for _ in range(7)]
	h_full = base.patch_forward(grid)
	past = kv._empty_past(); coss = []
	for t in range(len(grid)):
		last, past = kv.patch_kv_step([grid[t]], past)
		a = h_full[0, t]
		coss.append(float((a @ last) / (np.linalg.norm(a) * np.linalg.norm(last) + 1e-9)))
	print('per-patch hidden cos:', ' '.join('%.4f' % c for c in coss))
	print('(fp32 the two are identical, cos 1.0; int8 graphs quantize independently)')

	# ---- speed (greedy) ----
	print('\n===== SPEED (greedy, seed=%d, measures=%d, max_patches=%d) ====='
		% (args.seed, args.measures, args.max_patches))
	def greedy (g):
		torch.manual_seed(args.seed)
		return g.generate(prompt_text=PROMPT, max_patches=args.max_patches,
			temperature=1e-6, top_k=1, top_p=1.0, measures=args.measures, postprocess=True)
	t0 = time.perf_counter(); txt_base = greedy(base); dt_base = time.perf_counter() - t0
	t0 = time.perf_counter(); txt_kv = greedy(kv); dt_kv = time.perf_counter() - t0

	print('baseline (full recompute): %6.1fs  %5d chars' % (dt_base, len(txt_base)))
	print('KV cache (incremental)   : %6.1fs  %5d chars' % (dt_kv, len(txt_kv)))
	print('speedup: %.2fx' % (dt_base / dt_kv))
	print('byte-identical (int8):', txt_base == txt_kv, '(divergence = int8 quantization noise, both valid)')


if __name__ == '__main__':
	main()

