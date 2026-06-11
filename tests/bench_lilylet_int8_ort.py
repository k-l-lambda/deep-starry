"""INT8 + ONNX Runtime vs fp32 PyTorch, for the llama-lr0.2 LilyletNotaGen on CPU.

Pipeline:
  1. Load the torch model (architecture read from the run's .state.yaml).
  2. Export the two heavy transformer forwards to ONNX:
       - PatchNet : patch ids [1,T,patch_size] -> patch hidden states [1,T,hidden]
       - TokenNet : inputs_embeds [1,L,hidden]  -> logits [1,L,vocab]
     (the cheap one-hot/embedding lookups stay; only the transformers are exported)
  3. quantize_dynamic both to INT8 (weight int8, activation dynamic — no calibration).
  4. Build an ORT-int8 generator mirroring patchyGenerator.generate.
  5. Compare fp32-torch vs int8-ORT:
       - fidelity: cosine / max-abs-diff / top-1 agreement on identical inputs
       - latency : per char-token, per patch
       - end-to-end: greedy generation, diff the produced Lilylet text.

Usage:
  python tests/bench_lilylet_int8_ort.py --measures 8
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from starry.utils.config import Configuration
from starry.lilylet.patchyGenerator import LilyletPatchyGenerator, sample_next
from starry.lilylet.models.notagen import token_embedding_weight, PatchNet, TokenNet

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = '/home/camus/data/models/deep-starry-logs/lilylet/20260609-lilylet-notagenx-large-llama-lr0.2'
CKPT = os.path.join(RUN, 'best.chkpt')
ONNX_DIR = os.path.join(REPO_ROOT, 'tests', 'output', 'onnx')


def export_and_quantize (gen, hidden):
	'''Export PatchNet/TokenNet to ONNX and produce INT8 dynamic-quantized copies.'''
	import onnx  # noqa
	from onnxruntime.quantization import quantize_dynamic, QuantType

	os.makedirs(ONNX_DIR, exist_ok=True)
	model = gen.model
	patch_net = PatchNet(model).eval()
	token_net = TokenNet(model).eval()

	paths = {
		'patch_fp32': os.path.join(ONNX_DIR, 'patch_fp32.onnx'),
		'patch_int8': os.path.join(ONNX_DIR, 'patch_int8.onnx'),
		'token_fp32': os.path.join(ONNX_DIR, 'token_fp32.onnx'),
		'token_int8': os.path.join(ONNX_DIR, 'token_int8.onnx'),
	}

	# dummy inputs (dynamic axes let real T / L vary)
	dummy_patches = torch.randint(0, gen.tokenizer.vocab_size if hasattr(gen.tokenizer, 'vocab_size') else 256,
		(1, 4, gen.patch_size), dtype=torch.long)
	dummy_embed = torch.randn(1, 5, hidden)

	with torch.no_grad():
		torch.onnx.export(patch_net, (dummy_patches,), paths['patch_fp32'],
			input_names=['patches'], output_names=['hidden'],
			dynamic_axes={'patches': {1: 'T'}, 'hidden': {1: 'T'}}, opset_version=17)
		torch.onnx.export(token_net, (dummy_embed,), paths['token_fp32'],
			input_names=['inputs_embeds'], output_names=['logits'],
			dynamic_axes={'inputs_embeds': {1: 'L'}, 'logits': {1: 'L'}}, opset_version=17)

	print('exported fp32 onnx; quantizing to int8...')
	for lvl in ('patch', 'token'):
		quantize_dynamic(paths[f'{lvl}_fp32'], paths[f'{lvl}_int8'], weight_type=QuantType.QInt8)
	for k, p in paths.items():
		print('  %-11s %8.1f MB  %s' % (k, os.path.getsize(p) / 1e6, p))
	return paths


class ORTGenerator:
	'''Mirrors LilyletPatchyGenerator.generate but runs the two transformer
	forwards through ORT sessions. Embedding lookup + sampling stay in numpy/torch.'''
	def __init__ (self, gen, patch_onnx, token_onnx, threads=None):
		import onnxruntime as ort
		so = ort.SessionOptions()
		if threads:
			so.intra_op_num_threads = threads
		self.patch_sess = ort.InferenceSession(patch_onnx, so, providers=['CPUExecutionProvider'])
		self.token_sess = ort.InferenceSession(token_onnx, so, providers=['CPUExecutionProvider'])
		self.g = gen
		self.wte = token_embedding_weight(gen.model.token_level_decoder.base).detach().cpu().numpy()

	def patch_forward (self, patches_2d):
		x = np.asarray([patches_2d], dtype=np.int64)
		return self.patch_sess.run(None, {'patches': x})[0]  # [1,T,hidden]

	def token_logits (self, inputs_embeds_np):
		return self.token_sess.run(None, {'inputs_embeds': inputs_embeds_np.astype(np.float32)})[0]

	def generate_patch (self, last_hidden, prefix_ids=None, temperature=1.0, top_k=0, top_p=1.0):
		g = self.g
		tokens = [g.bos_id] + list(prefix_ids or [])
		generated = list(prefix_ids or [])
		enc = last_hidden.reshape(1, 1, -1)
		while len(generated) < g.patch_size:
			emb = self.wte[np.asarray(tokens)][None]          # [1,len,hidden]
			emb = np.concatenate([enc, emb[:, 1:, :]], axis=1)
			logits = self.token_logits(emb)[0, -1]
			nxt = sample_next(torch.from_numpy(logits), temperature=temperature, top_k=top_k, top_p=top_p)
			generated.append(nxt); tokens.append(nxt)
		return generated

	def generate (self, prompt_text='', max_patches=256, temperature=1.0, top_k=0, top_p=0.9,
		measures=None, postprocess=False):
		g = self.g
		bos_patch = [g.bos_id] * (g.patch_size - 1) + [g.eos_id]
		patches = [bos_patch]
		if prompt_text:
			for line in prompt_text.splitlines():
				ids = g.tokenizer.encode(line + '\n')
				for i in range(0, len(ids), g.patch_size):
					chunk = ids[i:i + g.patch_size]
					patches.append(chunk + [g.pad_id] * (g.patch_size - len(chunk)))
		out_text = g.patches_to_text(patches[1:])
		prime_ids = g.tokenizer.encode(f'[r:0/{measures}]') if measures is not None else None
		primed = False
		for _ in range(max_patches):
			flat = sum(patches, [])
			grid = [flat[i:i + g.patch_size] for i in range(0, len(flat), g.patch_size)]
			hidden = self.patch_forward(grid)
			last = hidden[0, -1]
			patch_ids = self.generate_patch(last, temperature=temperature, top_k=top_k, top_p=top_p)
			if prime_ids is not None and not primed and g.patch_to_text(patch_ids).startswith('[r:'):
				primed = True
				patch_ids = self.generate_patch(last, prefix_ids=prime_ids,
					temperature=temperature, top_k=top_k, top_p=top_p)
			if patch_ids[0] == g.bos_id and patch_ids[1] == g.eos_id:
				break
			out_text += g.patch_to_text(patch_ids)
			clean = list(patch_ids); seen = False
			for j in range(len(clean)):
				if seen: clean[j] = g.pad_id
				if clean[j] == g.eos_id: seen = True
			patches.append(clean)
		return g.postprocess(out_text) if postprocess else out_text


class ORTGeneratorKV (ORTGenerator):
	'''Patch-level KV-cache generator. Same token path as ORTGenerator, but the
	patch decoder runs incrementally through the patch_kv_int8 session instead of
	recomputing the whole patch sequence every step. Output is identical to
	ORTGenerator (greedy: byte-for-byte); only the patch-level cost changes from
	O(T) per step to O(1).'''

	def __init__ (self, gen, patch_kv_onnx, token_onnx, threads=None):
		import onnxruntime as ort
		so = ort.SessionOptions()
		if threads:
			so.intra_op_num_threads = threads
		self.patch_kv_sess = ort.InferenceSession(patch_kv_onnx, so, providers=['CPUExecutionProvider'])
		self.token_sess = ort.InferenceSession(token_onnx, so, providers=['CPUExecutionProvider'])
		self.g = gen
		self.wte = token_embedding_weight(gen.model.token_level_decoder.base).detach().cpu().numpy()
		# KV geometry from the patch-level base
		pbase = gen.model.patch_level_decoder.base
		self.n_layers = pbase.config.num_hidden_layers
		self.n_kv = pbase.config.num_key_value_heads
		self.head_dim = pbase.config.hidden_size // pbase.config.num_attention_heads
		self.out_names = [o.name for o in self.patch_kv_sess.get_outputs()]

	def _empty_past (self):
		return [np.zeros((1, self.n_kv, 0, self.head_dim), dtype=np.float32) for _ in range(2 * self.n_layers)]

	def patch_kv_step (self, patch_rows, past):
		'''Feed L new patches (list of patch_size-length id rows) + past KV.
		Returns (last_hidden [hidden], new_past list). new_past replaces past.'''
		x = np.asarray([patch_rows], dtype=np.int64)        # [1, L, patch_size]
		feed = {'patches': x}
		for i in range(self.n_layers):
			feed[f'past_k_{i}'] = past[2 * i]
			feed[f'past_v_{i}'] = past[2 * i + 1]
		out = dict(zip(self.out_names, self.patch_kv_sess.run(None, feed)))
		new_past = []
		for i in range(self.n_layers):
			new_past.append(out[f'new_k_{i}'])
			new_past.append(out[f'new_v_{i}'])
		return out['hidden'][0, -1], new_past

	def generate (self, prompt_text='', max_patches=256, temperature=1.0, top_k=0, top_p=0.9,
		measures=None, postprocess=False):
		g = self.g
		bos_patch = [g.bos_id] * (g.patch_size - 1) + [g.eos_id]
		patches = [bos_patch]
		if prompt_text:
			for line in prompt_text.splitlines():
				ids = g.tokenizer.encode(line + '\n')
				for i in range(0, len(ids), g.patch_size):
					chunk = ids[i:i + g.patch_size]
					patches.append(chunk + [g.pad_id] * (g.patch_size - len(chunk)))
		out_text = g.patches_to_text(patches[1:])
		prime_ids = g.tokenizer.encode(f'[r:0/{measures}]') if measures is not None else None
		primed = False

		# prefill: run all seed patches through the KV decoder in one call
		past = self._empty_past()
		last, past = self.patch_kv_step(patches, past)

		for _ in range(max_patches):
			patch_ids = self.generate_patch(last, temperature=temperature, top_k=top_k, top_p=top_p)
			if prime_ids is not None and not primed and g.patch_to_text(patch_ids).startswith('[r:'):
				primed = True
				patch_ids = self.generate_patch(last, prefix_ids=prime_ids,
					temperature=temperature, top_k=top_k, top_p=top_p)
			if patch_ids[0] == g.bos_id and patch_ids[1] == g.eos_id:
				break
			out_text += g.patch_to_text(patch_ids)
			clean = list(patch_ids); seen = False
			for j in range(len(clean)):
				if seen: clean[j] = g.pad_id
				if clean[j] == g.eos_id: seen = True
			# advance the patch-level cache by the one new patch -> next hidden state
			last, past = self.patch_kv_step([clean], past)
		return g.postprocess(out_text) if postprocess else out_text


def _cos (a, b):
	a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
	return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--measures', type=int, default=8)
	ap.add_argument('--max-patches', type=int, default=64)
	ap.add_argument('--threads', type=int, default=14)
	ap.add_argument('--seed', type=int, default=0)
	args = ap.parse_args()

	torch.set_num_threads(args.threads)
	config = Configuration.createOrLoad(RUN, volatile=True)
	tk = config['data.args.tokenizer_path']
	tk = tk if os.path.isabs(tk) else os.path.join(REPO_ROOT, tk)
	gen = LilyletPatchyGenerator.from_config(config, CKPT, tokenizer_path=tk, device='cpu')
	hidden = config['model.args.hidden_size']
	print('params: %.1fM | hidden %d | threads %d' %
		(sum(p.numel() for p in gen.model.parameters()) / 1e6, hidden, args.threads))

	paths = export_and_quantize(gen, hidden)
	ort_i8 = ORTGenerator(gen, paths['patch_int8'], paths['token_int8'], threads=args.threads)
	ort_f32 = ORTGenerator(gen, paths['patch_fp32'], paths['token_fp32'], threads=args.threads)

	# ---- fidelity: same inputs through torch-fp32 vs ORT-int8 ----
	print('\n===== FIDELITY (identical inputs) =====')
	demo = gen.tokenizer.encode('[r:0/8]\\staff "1" \\key c \\major')[:gen.patch_size]
	demo = demo + [gen.pad_id] * (gen.patch_size - len(demo))
	grid = [[gen.bos_id] * (gen.patch_size - 1) + [gen.eos_id], demo]
	with torch.no_grad():
		t_hidden = gen.model.patch_level_decoder(torch.tensor([sum(grid, [])]).reshape(1, -1, gen.patch_size))['last_hidden_state'].numpy()
	i_hidden = ort_i8.patch_forward(grid)
	print('patch hidden: cos %.5f  max|Δ| %.4f' % (_cos(t_hidden, i_hidden), np.abs(t_hidden - i_hidden).max()))

	last_t = torch.from_numpy(t_hidden[0, -1])
	emb = F.embedding(torch.tensor([gen.bos_id, demo[0], demo[1]]), token_embedding_weight(gen.model.token_level_decoder.base))
	emb = torch.cat((last_t.reshape(1, 1, -1), emb[1:].reshape(1, -1, hidden)), dim=1).detach()
	with torch.no_grad():
		t_logits = gen.model.token_level_decoder.base(inputs_embeds=emb).logits.numpy()
	i_logits = ort_i8.token_logits(emb.numpy())
	top1 = (t_logits[0].argmax(-1) == i_logits[0].argmax(-1)).mean()
	print('char logits:  cos %.5f  max|Δ| %.4f  top1-agree %.3f' %
		(_cos(t_logits, i_logits), np.abs(t_logits - i_logits).max(), top1))

	# ---- latency: per char-token / per patch ----
	print('\n===== LATENCY (greedy, measures=%d, max_patches=%d) =====' % (args.measures, args.max_patches))
	def timed (label, fn):
		t0 = time.perf_counter(); txt = fn(); dt = time.perf_counter() - t0
		nl = txt.count('\n') + 1
		print('%-12s %6.1fs  %5d chars  %4d lines' % (label, dt, len(txt), nl))
		return txt, dt

	torch.manual_seed(args.seed)
	txt_torch, dt_torch = timed('torch-fp32', lambda: gen.generate(
		max_patches=args.max_patches, temperature=1e-6, top_k=1, top_p=1.0, measures=args.measures, postprocess=True))
	torch.manual_seed(args.seed)
	txt_ortf, dt_ortf = timed('ORT-fp32', lambda: ort_f32.generate(
		max_patches=args.max_patches, temperature=1e-6, top_k=1, top_p=1.0, measures=args.measures, postprocess=True))
	torch.manual_seed(args.seed)
	txt_i8, dt_i8 = timed('ORT-int8', lambda: ort_i8.generate(
		max_patches=args.max_patches, temperature=1e-6, top_k=1, top_p=1.0, measures=args.measures, postprocess=True))

	print('\nspeedup vs torch-fp32:  ORT-fp32 %.2fx  ORT-int8 %.2fx' % (dt_torch / dt_ortf, dt_torch / dt_i8))

	# ---- end-to-end output diff ----
	print('\n===== OUTPUT MATCH (greedy) =====')
	print('torch-fp32 == ORT-fp32 :', txt_torch == txt_ortf)
	print('torch-fp32 == ORT-int8 :', txt_torch == txt_i8)
	for name, txt in [('torch_fp32', txt_torch), ('ort_int8', txt_i8)]:
		out = os.path.join(REPO_ROOT, 'tests', 'output', f'lilylet_seed{args.seed}_{name}.lyl')
		with open(out, 'w') as f:
			f.write(txt)
		print('wrote', out)


if __name__ == '__main__':
	main()



