"""Generate Lilylet from a prompt using the already-exported INT8 ONNX weights.

Reuses ORTGenerator from bench_lilylet_int8_ort.py and the int8 onnx artifacts
in tests/output/onnx/. No re-export/re-quantization.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from starry.utils.config import Configuration
from starry.lilylet.patchyGenerator import LilyletPatchyGenerator
from starry.lilylet.mask_monitor import MaskMonitor, load_blacklist
from bench_lilylet_int8_ort import ORTGenerator, RUN, CKPT, ONNX_DIR, REPO_ROOT


PROMPT = '%Romantic\n%Schubert, Franz\n%Keyboard\n'
DEFAULT_BLACKLIST = '/home/camus/work/LilyScript/assets/lilylet-blacklist.json'


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
	ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'tests', 'output', 'lilylet_schubert_int8.lyl'))
	args = ap.parse_args()

	torch.set_num_threads(args.threads)
	torch.manual_seed(args.seed)

	config = Configuration.createOrLoad(RUN, volatile=True)
	tk = config['data.args.tokenizer_path']
	tk = tk if os.path.isabs(tk) else os.path.join(REPO_ROOT, tk)
	gen = LilyletPatchyGenerator.from_config(config, CKPT, tokenizer_path=tk, device='cpu')

	patch_i8 = os.path.join(ONNX_DIR, 'patch_int8.onnx')
	token_i8 = os.path.join(ONNX_DIR, 'token_int8.onnx')
	assert os.path.isfile(patch_i8) and os.path.isfile(token_i8), 'run bench_lilylet_int8_ort.py first to export int8 onnx'
	ort_i8 = ORTGenerator(gen, patch_i8, token_i8, threads=args.threads)

	monitor = None
	if args.blacklist:
		blacklist = load_blacklist(args.blacklist)
		if blacklist:
			monitor = MaskMonitor(gen, blacklist)
			print('syntax blacklist: %d context keys from %s' % (len(blacklist), args.blacklist))
		else:
			print('syntax blacklist: %s empty/missing -> masking disabled' % args.blacklist)

	print('=== INT8 ONNX generation ===')
	print('prompt:')
	print(PROMPT)
	print('--- output (seed=%d temp=%.2f top_k=%d top_p=%.2f measures=%s blacklist=%s) ---\n'
		% (args.seed, args.temperature, args.top_k, args.top_p, args.measures, monitor is not None))

	t0 = time.perf_counter()
	text = ort_i8.generate(prompt_text=PROMPT, max_patches=args.max_patches,
		temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
		measures=args.measures, postprocess=True, monitor=monitor)
	dt = time.perf_counter() - t0

	print(text)
	nl = text.count('\n') + 1
	print('\n===== %d chars, %d lines in %.1fs =====' % (len(text), nl, dt))
	with open(args.out, 'w') as f:
		f.write(text)
	print('wrote', args.out)


if __name__ == '__main__':
	main()
