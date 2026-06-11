"""CPU inference latency benchmark for the llama-lr0.2 LilyletNotaGen.

Measures, on CPU, the wall-clock cost of autoregressive decoding for the
hierarchical patch/char model:
  - per char-level token (one char-decoder forward over the growing patch)
  - per patch            (one patch-decoder forward + patch_size char tokens)

Architecture is read from the checkpoint run's own .state.yaml, so it always
matches the weights.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from starry.utils.config import Configuration
from starry.lilylet.patchyGenerator import LilyletPatchyGenerator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = '/home/camus/data/models/deep-starry-logs/lilylet/20260609-lilylet-notagenx-large-llama-lr0.2'
CKPT = os.path.join(RUN, 'best.chkpt')


def main ():
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument('--patches', type=int, default=8, help='patches to generate for timing')
	ap.add_argument('--threads', type=int, default=None, help='torch CPU threads (default: torch default)')
	ap.add_argument('--warmup-patches', type=int, default=2, help='patches to generate before timing (warmup)')
	args = ap.parse_args()

	if args.threads:
		torch.set_num_threads(args.threads)
	print('torch threads:', torch.get_num_threads())

	config = Configuration.createOrLoad(RUN, volatile=True)
	tk = config['data.args.tokenizer_path']
	tk = tk if os.path.isabs(tk) else os.path.join(REPO_ROOT, tk)
	print('base_type:', config['model.args.base_type'], '| building on CPU...')

	gen = LilyletPatchyGenerator.from_config(config, CKPT, tokenizer_path=tk, device='cpu')
	n_params = sum(p.numel() for p in gen.model.parameters())
	print('params: %.2fM | patch_size: %d | device: %s' % (n_params / 1e6, gen.patch_size, gen.device))

	# Instrument the two forward levels by timing a real generation run. We
	# replicate the generate() inner loop here so we can time each level.
	patch_dec = gen.model.patch_level_decoder
	char_base = gen.model.token_level_decoder.base
	from starry.lilylet.models.notagen import token_embedding_weight
	import torch.nn.functional as F
	wte = token_embedding_weight(char_base)

	bos_patch = [gen.bos_id] * (gen.patch_size - 1) + [gen.eos_id]
	patches = [bos_patch]

	patch_times = []
	char_token_times = []

	torch.manual_seed(0)
	total_patches = args.warmup_patches + args.patches
	with torch.no_grad():
		for p_i in range(total_patches):
			timing = p_i >= args.warmup_patches

			# ---- patch-level forward ----
			t0 = time.perf_counter()
			inp = torch.tensor([sum(patches, [])]).reshape(1, -1, gen.patch_size)
			encoded = patch_dec(inp)['last_hidden_state']
			last = encoded[0, -1]
			t1 = time.perf_counter()
			if timing:
				patch_fwd = t1 - t0

			# ---- char-level: sample patch_size tokens ----
			tokens = [gen.bos_id]
			generated = []
			enc = last.reshape(1, 1, -1)
			while len(generated) < gen.patch_size:
				tc0 = time.perf_counter()
				tok_tensor = torch.tensor([tokens])
				emb = F.embedding(tok_tensor, wte)
				emb = torch.cat((enc, emb[:, 1:, :]), dim=1)
				logits = char_base(inputs_embeds=emb).logits[0, -1]
				nxt = int(torch.argmax(logits))
				tc1 = time.perf_counter()
				if timing:
					char_token_times.append(tc1 - tc0)
				generated.append(nxt)
				tokens.append(nxt)

			if timing:
				patch_times.append(patch_fwd + sum(char_token_times[-gen.patch_size:]))
			patches.append(generated)
			ctx = len(patches)
			print('patch %2d | ctx=%3d patches | patch-fwd %.3fs%s'
				% (p_i, ctx, (t1 - t0), '' if timing else '  (warmup)'))

	import statistics as st
	print('\n===== CPU inference timing (%d timed patches, ctx grows %d->%d) ====='
		% (args.patches, args.warmup_patches + 1, total_patches))
	ct = char_token_times
	print('char-level token:  mean %.4fs  median %.4fs  min %.4fs  max %.4fs  (n=%d)'
		% (st.mean(ct), st.median(ct), min(ct), max(ct), len(ct)))
	print('per full patch:    mean %.3fs  median %.3fs  (= 1 patch-fwd + %d char tokens)'
		% (st.mean(patch_times), st.median(patch_times), gen.patch_size))
	print('throughput:        %.2f char-tokens/s | %.3f patches/s'
		% (1.0 / st.mean(ct), 1.0 / st.mean(patch_times)))


if __name__ == '__main__':
	main()
