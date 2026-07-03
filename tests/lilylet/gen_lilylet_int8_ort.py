"""Generate Lilylet from a prompt using the already-exported INT8 ONNX KV-cache weights.

Reuses ORTGeneratorKV from bench_lilylet_int8_ort.py and the int8 KV onnx artifacts
(patch_kv_int8.onnx, token_kv_int8.onnx) in tests/output/onnx/. The KV-cache variant
runs incremental patch-level + token-level decoding (O(1) per step, like LilyScript's
standalone StreamingLilyletGenerator). No re-export/re-quantization.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from starry.utils.config import Configuration
from starry.lilylet.patchyGenerator import LilyletPatchyGenerator
from starry.lilylet.mask_monitor import MaskMonitor, load_blacklist
from bench_lilylet_int8_ort import ORTGeneratorKV, RUN, CKPT, ONNX_DIR, REPO_ROOT


PROMPT = '%Romantic\n%Schubert, Franz\n%Keyboard\n'
DEFAULT_BLACKLIST = '../LilyScript/assets/lilylet-blacklist.json'


def main ():
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument('--measures', type=int, default=None)
	ap.add_argument('--max-patches', type=int, default=1024)
	ap.add_argument('--threads', type=int, default=14)
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--temperature', type=float, default=0.9)
	ap.add_argument('--top-k', type=int, default=20)
	ap.add_argument('--top-p', type=float, default=0.95)
	ap.add_argument('--blacklist', default=DEFAULT_BLACKLIST,
		help='syntax-blacklist JSON path; pass "" to disable masking')
	ap.add_argument('--run', default=RUN,
		help='training run dir (holds best.chkpt + onnx/); default: the bench RUN')
	ap.add_argument('--num-samples', type=int, default=1,
		help='number of samples to generate, one per consecutive seed starting at --seed')
	ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'tests', 'output', 'lilylet_schubert_int8.lyl'))
	args = ap.parse_args()

	torch.set_num_threads(args.threads)

	run = args.run
	ckpt = os.path.join(run, 'best.chkpt')
	onnx_dir = os.path.join(run, 'onnx')

	config = Configuration.createOrLoad(run, volatile=True)
	tk = config['data.args.tokenizer_path']
	tk = tk if os.path.isabs(tk) else os.path.join(REPO_ROOT, tk)
	gen = LilyletPatchyGenerator.from_config(config, ckpt, tokenizer_path=tk, device='cpu')

	patch_kv_i8 = os.path.join(onnx_dir, 'patch_kv_int8.onnx')
	token_kv_i8 = os.path.join(onnx_dir, 'token_kv_int8.onnx')
	token_i8 = os.path.join(onnx_dir, 'token_int8.onnx')
	assert os.path.isfile(patch_kv_i8) and os.path.isfile(token_kv_i8) and os.path.isfile(token_i8), \
		'run tools/lilylet/export_lilylet_int8_ort.py first to export KV int8 onnx'
	# ORTGeneratorKV with both patch-level AND token-level KV cache (both incremental,
	# O(1) per step). token_i8 is the non-KV token fallback session; token_kv_onnx is
	# the incremental one actually used. Mirrors LilyScript's StreamingLilyletGenerator.
	ort_kv = ORTGeneratorKV(gen, patch_kv_i8, token_i8, threads=args.threads, token_kv_onnx=token_kv_i8)

	blacklist = load_blacklist(args.blacklist) if args.blacklist else None
	use_mask = bool(blacklist)
	if args.blacklist and not blacklist:
		print('syntax blacklist: %s empty/missing -> masking disabled' % args.blacklist)
	elif use_mask:
		print('syntax blacklist: %d context keys from %s' % (len(blacklist), args.blacklist))

	print('=== INT8 ONNX KV-cache generation (patch+token incremental) ===')
	print('run:', run)
	print('prompt:')
	print(PROMPT)
	print('--- %d sample(s), seeds %d..%d (temp=%.2f top_k=%d top_p=%.2f measures=%s blacklist=%s) ---'
		% (args.num_samples, args.seed, args.seed + args.num_samples - 1,
			args.temperature, args.top_k, args.top_p, args.measures, use_mask))

	# output path: for a single sample keep --out as-is; for many, insert the seed
	# before the extension (…_int8.lyl -> …_int8.seedN.lyl) so samples don't overwrite.
	out_base, out_ext = os.path.splitext(args.out)

	for i in range(args.num_samples):
		seed = args.seed + i
		torch.manual_seed(seed)
		# fresh monitor per sample: MaskMonitor carries running context/stream state,
		# so reusing one across samples would leak the previous sample's tail.
		monitor = MaskMonitor(gen, blacklist) if use_mask else None

		t0 = time.perf_counter()
		text = ort_kv.generate(prompt_text=PROMPT, max_patches=args.max_patches,
			temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
			measures=args.measures, postprocess=True, monitor=monitor)
		dt = time.perf_counter() - t0

		out = args.out if args.num_samples == 1 else '%s.seed%d%s' % (out_base, seed, out_ext)
		with open(out, 'w') as f:
			f.write(text)
		nl = text.count('\n') + 1
		print('\n===== seed %d: %d chars, %d lines in %.1fs -> %s =====' % (seed, len(text), nl, dt, out))
		print(text)


if __name__ == '__main__':
	main()
