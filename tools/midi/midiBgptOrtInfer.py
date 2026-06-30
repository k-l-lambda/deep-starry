"""MidiBGPT inference via INT8 ONNX-Runtime with a two-level KV cache.

Self-contained tool: loads a trained MidiBGPT checkpoint, exports the two-level
bGPT decoders to INT8 ONNX (reusing the modality-agnostic KV-net wrappers in
starry.bgpt.kv_net), then runs incremental KV-cache generation and decodes the
result back to MidiText (one event per line — the same format the model trained on).

The heavy transformer forwards run through ORT INT8 sessions; the cheap pieces
(one-hot is inside the patch graph, token embedding lookup + sampling stay outside)
run in numpy/torch. Four graphs are exported:
  - patch_int8.onnx    PatchNet     patch ids -> patch hidden states (full recompute)
  - token_int8.onnx    TokenNet     inputs_embeds -> logits
  - patch_kv_int8.onnx PatchNetKV   patch ids + past KV -> hidden + present KV (incremental)
  - token_kv_int8.onnx TokenNetKV   inputs_embeds + past KV -> logits + present KV (incremental)

Architecture is read from the run's own .state.yaml so it always matches the weights.

Usage:
  python tools/midi/midiBgptOrtInfer.py \
    --run /home/camus/data/models/deep-starry-logs/midi/20260624-midi-bgpt-patchy0624-l4+10 \
    --max-patches 512 --temperature 1.0 --top-p 0.9
"""

import os
import sys
import time
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModel
from starry.midi.tokenizer import MidiTokenizer
from starry.bgpt import PatchNet, TokenNet, PatchNetKV, TokenNetKV, token_embedding_weight
from starry.lilylet.patchyGenerator import sample_next


def _rm (path):
	for p in (path, path + '.data'):
		if os.path.exists(p):
			os.remove(p)


def load_model (run, checkpoint):
	'''Build MidiBGPT from the run's config (.state.yaml) and load weights (CPU, eval).'''
	config = Configuration.createOrLoad(run, volatile=True)
	model = loadModel(config['model'], imports=config['imports'])
	ckpt = torch.load(checkpoint, map_location='cpu')
	state = ckpt['model'] if 'model' in ckpt else ckpt
	missing, unexpected = model.load_state_dict(state, strict=False)
	if missing or unexpected:
		print(f'[load] {len(missing)} missing, {len(unexpected)} unexpected keys')
	model.eval()
	return config, model


def export_int8 (model, hidden, vocab, patch_size, out_dir):
	'''Export PatchNet/TokenNet (full recompute) to fp32 ONNX, quantize to int8.'''
	from onnxruntime.quantization import quantize_dynamic, QuantType

	os.makedirs(out_dir, exist_ok=True)
	patch_net = PatchNet(model).eval()
	token_net = TokenNet(model).eval()

	# dummy ids avoid the special tokens (0..3); real T / L vary via dynamic axes.
	dummy_patches = torch.randint(4, vocab, (1, 4, patch_size), dtype=torch.long)
	dummy_embed = torch.randn(1, 5, hidden)

	specs = {
		'patch': (patch_net, (dummy_patches,), ['patches'], ['hidden'],
			{'patches': {1: 'T'}, 'hidden': {1: 'T'}}),
		'token': (token_net, (dummy_embed,), ['inputs_embeds'], ['logits'],
			{'inputs_embeds': {1: 'L'}, 'logits': {1: 'L'}}),
	}
	results = {}
	for name, (net, dummy, inames, onames, dyn) in specs.items():
		fp32 = os.path.join(out_dir, f'{name}_fp32_tmp.onnx')
		int8 = os.path.join(out_dir, f'{name}_int8.onnx')
		with torch.no_grad():
			torch.onnx.export(net, dummy, fp32, input_names=inames, output_names=onames,
				dynamic_axes=dyn, opset_version=17)
		quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
		_rm(fp32)
		results[f'{name}_int8'] = int8
	return results


def _export_kv (net, base, dummy_new, out_dir, name, vocab, patch_size):
	'''Shared KV-graph export: dynamo + dynamic_shapes (L=Dim.AUTO, P=Dim('P')).'''
	from onnxruntime.quantization import quantize_dynamic, QuantType
	from torch.export import Dim

	NL = base.config.num_hidden_layers
	NKV = base.config.num_key_value_heads
	HD = base.config.hidden_size // base.config.num_attention_heads
	dummy_past = [torch.randn(1, NKV, 3, HD) for _ in range(2 * NL)]

	# L (new-step count) uses Dim.AUTO: a full-MHA model routes attention through the
	# plain matmul decomposition, which adds a benign `Ne(L, 1)` guard; an explicit
	# Dim('L', min=1) collides with it. Dim.AUTO keeps the axis dynamic (ONNX MatMul is
	# shape-polymorphic, so L=1 still runs). P (cache length) carries no such guard.
	L = Dim.AUTO
	P = Dim('P', min=1, max=8192)
	dynamic_shapes = ({1: L}, [{2: P} for _ in range(2 * NL)])
	in_names = [('patches' if name == 'patch_kv' else 'inputs_embeds')] \
		+ sum([[f'past_k_{i}', f'past_v_{i}'] for i in range(NL)], [])
	out_names = [('hidden' if name == 'patch_kv' else 'logits')] \
		+ sum([[f'new_k_{i}', f'new_v_{i}'] for i in range(NL)], [])

	fp32 = os.path.join(out_dir, f'{name}_fp32_tmp.onnx')
	int8 = os.path.join(out_dir, f'{name}_int8.onnx')
	with torch.no_grad():
		torch.onnx.export(net, (dummy_new, dummy_past), fp32,
			input_names=in_names, output_names=out_names,
			dynamic_shapes=dynamic_shapes, opset_version=18, dynamo=True)
	quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
	_rm(fp32)
	return {f'{name}_int8': int8}


def export_kv_int8 (model, hidden, vocab, patch_size, out_dir):
	'''Export PatchNetKV + TokenNetKV (incremental KV) to int8 ONNX.'''
	os.makedirs(out_dir, exist_ok=True)
	res = {}
	patch_kv = PatchNetKV(model).eval()
	dummy_patches = torch.randint(4, vocab, (1, 2, patch_size), dtype=torch.long)
	res.update(_export_kv(patch_kv, model.patch_level_decoder.base, dummy_patches,
		out_dir, 'patch_kv', vocab, patch_size))
	token_kv = TokenNetKV(model).eval()
	dummy_embed = torch.randn(1, 2, hidden)
	res.update(_export_kv(token_kv, model.token_level_decoder.base, dummy_embed,
		out_dir, 'token_kv', vocab, patch_size))
	return res


def ensure_export (model, hidden, vocab, patch_size, out_dir, reexport=False):
	'''Export all four int8 graphs unless they already exist.'''
	needed = ['patch_int8', 'token_int8', 'patch_kv_int8', 'token_kv_int8']
	paths = {k: os.path.join(out_dir, f'{k}.onnx') for k in needed}
	if not reexport and all(os.path.exists(p) for p in paths.values()):
		print('[export] all int8 graphs present, skipping (use --reexport to force)')
		return paths
	t0 = time.perf_counter()
	paths.update(export_int8(model, hidden, vocab, patch_size, out_dir))
	paths.update(export_kv_int8(model, hidden, vocab, patch_size, out_dir))
	print('[export] done in %.1fs' % (time.perf_counter() - t0))
	for k in needed:
		print('  %-16s %8.1f MB' % (k, os.path.getsize(paths[k]) / 1e6))
	return paths


class MidiOrtKVGenerator:
	'''Two-level KV-cache generator for MidiBGPT, running the int8 ONNX graphs.

	Patch level: feed new patches incrementally through patch_kv_int8 (O(1) per step
	vs O(T) full recompute). Token level: inside each patch, decode patch_size tokens
	through token_kv_int8 (position 0 = patch hidden state, positions 1.. = token
	embeddings). The token-embedding lookup + sampling stay in numpy/torch.
	'''

	def __init__ (self, model, tokenizer, onnx_paths, threads=None):
		import onnxruntime as ort
		so = ort.SessionOptions()
		if threads:
			so.intra_op_num_threads = threads
		prov = ['CPUExecutionProvider']
		self.patch_kv_sess = ort.InferenceSession(onnx_paths['patch_kv_int8'], so, providers=prov)
		self.token_kv_sess = ort.InferenceSession(onnx_paths['token_kv_int8'], so, providers=prov)
		self.tk = tokenizer
		self.patch_size = model.patch_size
		self.pad_id = model.special_token_id
		self.bos_id = model.bos_token_id
		self.eos_id = model.eos_token_id
		self.wte = token_embedding_weight(model.token_level_decoder.base).detach().cpu().numpy()

		pbase = model.patch_level_decoder.base.config
		self.n_layers = pbase.num_hidden_layers
		self.n_kv = pbase.num_key_value_heads
		self.head_dim = pbase.hidden_size // pbase.num_attention_heads
		self.patch_out_names = [o.name for o in self.patch_kv_sess.get_outputs()]

		tbase = model.token_level_decoder.base.config
		self.t_layers = tbase.num_hidden_layers
		self.t_kv = tbase.num_key_value_heads
		self.t_head_dim = tbase.hidden_size // tbase.num_attention_heads
		self.token_out_names = [o.name for o in self.token_kv_sess.get_outputs()]

	# --- patch-level KV ---
	def _empty_patch_past (self):
		return [np.zeros((1, self.n_kv, 0, self.head_dim), dtype=np.float32) for _ in range(2 * self.n_layers)]

	def patch_kv_step (self, patch_rows, past):
		'''Feed L new patches (list of patch_size-long id rows) + past KV.
		Returns (last_hidden [hidden], new_past).'''
		x = np.asarray([patch_rows], dtype=np.int64)        # [1, L, patch_size]
		feed = {'patches': x}
		for i in range(self.n_layers):
			feed[f'past_k_{i}'] = past[2 * i]
			feed[f'past_v_{i}'] = past[2 * i + 1]
		out = dict(zip(self.patch_out_names, self.patch_kv_sess.run(None, feed)))
		new_past = []
		for i in range(self.n_layers):
			new_past.append(out[f'new_k_{i}'])
			new_past.append(out[f'new_v_{i}'])
		return out['hidden'][0, -1], new_past

	# --- token-level KV ---
	def _empty_token_past (self):
		return [np.zeros((1, self.t_kv, 0, self.t_head_dim), dtype=np.float32) for _ in range(2 * self.t_layers)]

	def _token_kv_step (self, emb_np, past):
		'''Feed L new token embeddings [1,L,hidden] + past KV. Returns (logits[-1], new_past).'''
		feed = {'inputs_embeds': emb_np.astype(np.float32)}
		for i in range(self.t_layers):
			feed[f'past_k_{i}'] = past[2 * i]
			feed[f'past_v_{i}'] = past[2 * i + 1]
		out = dict(zip(self.token_out_names, self.token_kv_sess.run(None, feed)))
		new_past = []
		for i in range(self.t_layers):
			new_past.append(out[f'new_k_{i}'])
			new_past.append(out[f'new_v_{i}'])
		return out['logits'][0, -1], new_past

	def generate_patch (self, last_hidden, temperature=1.0, top_k=0, top_p=1.0):
		'''Decode one event patch, conditioned on the patch hidden state.

		MIDI event patches are terminated by an in-patch <eos> (then padded), mirroring the
		lilylet patchifier's chunk-ending convention. Stop token-level decoding at that
		<eos> and right-pad the rest; if no <eos> is produced, fall back to a full patch.
		'''
		generated = []
		past = self._empty_token_past()
		# position 0: patch hidden state (not an embedding lookup)
		enc = last_hidden.reshape(1, 1, -1).astype(np.float32)
		logits, past = self._token_kv_step(enc, past)
		while len(generated) < self.patch_size:
			nxt = sample_next(torch.from_numpy(logits), temperature=temperature, top_k=top_k, top_p=top_p)
			generated.append(nxt)
			if nxt == self.eos_id:
				generated += [self.pad_id] * (self.patch_size - len(generated))
				break
			if len(generated) >= self.patch_size:
				break
			emb = self.wte[nxt].reshape(1, 1, -1)
			logits, past = self._token_kv_step(emb, past)
		return generated

	def _seed_patches (self, prompt_text=''):
		'''Build the seed patch list: optional header event patches, then the <bos> patch.
		Header lines (ticks_per_beat / format_type / track …) are encoded as event patches;
		the <bos> patch marks the start of generation. With no prompt this is just [<bos>].'''
		patches = []
		if prompt_text.strip():
			enc, _ = self.tk.encode_patches(prompt_text, add_special_patches=False)
			patches.extend(enc)
		patches.append(self.tk.special_patch('bos'))
		return patches

	def generate (self, prompt_text='', max_patches=512, temperature=1.0, top_k=0, top_p=0.9, verbose=False):
		'''Autoregressively generate a MidiText document and return the decoded text.'''
		patches = self._seed_patches(prompt_text)
		# prefill: run all seed patches through the patch KV decoder in one call
		past = self._empty_patch_past()
		last, past = self.patch_kv_step(patches, past)

		body = []
		for _ in range(max_patches):
			patch_ids = self.generate_patch(last, temperature=temperature, top_k=top_k, top_p=top_p)
			# EOS patch [bos, eos, ...] -> stop
			if patch_ids[0] == self.bos_id and patch_ids[1] == self.eos_id:
				if verbose:
					print('[EOS patch -> stop]')
				break
			# mask tokens after the first in-patch EOS to PAD before appending
			clean = list(patch_ids)
			seen = False
			for j in range(len(clean)):
				if seen:
					clean[j] = self.pad_id
				if clean[j] == self.eos_id:
					seen = True
			body.append(clean)
			if verbose:
				line = self.tk.decode_event(clean)
				if line:
					print(line)
			# advance the patch-level cache by the one new patch -> next hidden state
			last, past = self.patch_kv_step([clean], past)
		return self.tk.decode_patches(body)


def _cos (a, b):
	a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
	return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fidelity_check (model, gen, tokenizer):
	'''Compare torch-fp32 patch hidden / token logits against the int8 KV graphs on
	identical inputs (cos + top-1 agreement) to confirm the exported graphs are correct.'''
	ps = tokenizer.patch_size
	rng = np.random.default_rng(0)
	grid = [tokenizer.special_patch('bos')] + [list(rng.integers(4, model.token_vocab_size, ps)) for _ in range(5)]

	with torch.no_grad():
		t_hidden = model.patch_level_decoder(
			torch.tensor([sum(grid, [])]).reshape(1, -1, ps))['last_hidden_state'].numpy()
	# int8 KV: feed the same patches, compare the last patch's hidden state
	past = gen._empty_patch_past()
	i_last, _ = gen.patch_kv_step(grid, past)
	print('patch hidden (last): cos %.5f' % _cos(t_hidden[0, -1], i_last))

	# token logits: position 0 = patch state, then a couple of token embeddings
	last_t = torch.from_numpy(t_hidden[0, -1])
	wte = token_embedding_weight(model.token_level_decoder.base)
	toks = [model.bos_token_id, grid[-1][0], grid[-1][1]]
	emb = torch.nn.functional.embedding(torch.tensor(toks), wte)
	emb = torch.cat((last_t.reshape(1, 1, -1), emb[1:].reshape(1, -1, wte.shape[1])), dim=1).detach()
	with torch.no_grad():
		t_logits = model.token_level_decoder.base(inputs_embeds=emb).logits.numpy()
	# int8 KV token path: feed the same 3 positions incrementally
	tpast = gen._empty_token_past()
	il, tpast = gen._token_kv_step(emb[:, 0:1, :].numpy(), tpast)
	for k in (1, 2):
		il, tpast = gen._token_kv_step(emb[:, k:k + 1, :].numpy(), tpast)
	top1 = int(t_logits[0, -1].argmax() == il.argmax())
	print('token logits (last): cos %.5f  top1-match %d' % (_cos(t_logits[0, -1], il), top1))


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--run', required=True, help='training run dir (holds best.chkpt + .state.yaml)')
	ap.add_argument('--checkpoint', default=None, help='checkpoint path (default: <run>/best.chkpt)')
	ap.add_argument('--output-dir', default=None, help='onnx output dir (default: <run>/onnx_midi)')
	ap.add_argument('--tokenizer', default=None, help='tokenizer.json path (default: <run>/tokenizer.json, else in-code build)')
	ap.add_argument('--prompt-file', default=None, help='optional MidiText header lines as seed prefix')
	ap.add_argument('--max-patches', type=int, default=512)
	ap.add_argument('--temperature', type=float, default=1.0)
	ap.add_argument('--top-k', type=int, default=0)
	ap.add_argument('--top-p', type=float, default=0.9)
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--threads', type=int, default=14)
	ap.add_argument('--reexport', action='store_true', help='force re-export even if onnx exists')
	ap.add_argument('--fidelity', action='store_true', help='run the int8-vs-fp32 fidelity check')
	ap.add_argument('--output', default=None, help='write generated MidiText to this file')
	args = ap.parse_args()

	torch.set_num_threads(args.threads)
	torch.manual_seed(args.seed)

	run = args.run
	ckpt = args.checkpoint or os.path.join(run, 'best.chkpt')
	out_dir = args.output_dir or os.path.join(run, 'onnx_midi')
	print('run:', run, '| checkpoint:', ckpt, '| onnx:', out_dir)

	config, model = load_model(run, ckpt)
	hidden = config['model.args.hidden_size']
	patch_size = config['model.args.patch_size']
	# Prefer a pinned tokenizer.json (run dir or --tokenizer) so the inference vocab is
	# exactly the one frozen at training time; fall back to the in-code deterministic build.
	tok_path = args.tokenizer or os.path.join(run, 'tokenizer.json')
	if os.path.exists(tok_path):
		tokenizer = MidiTokenizer.from_json(tok_path)
		print('[tokenizer] loaded', tok_path, '| vocab:', tokenizer.vocab_size)
		if tokenizer.patch_size != patch_size:
			print('[tokenizer] WARN patch_size %d != config %d; using config'
				% (tokenizer.patch_size, patch_size))
			tokenizer.patch_size = patch_size
	else:
		tokenizer = MidiTokenizer(patch_size)
		print('[tokenizer] built in-code (no tokenizer.json at %s)' % tok_path)
	vocab = config['model.args.token_vocab_size']
	n_params = sum(p.numel() for p in model.parameters())
	print('params: %.2fM | base: %s | hidden: %d | patch_size: %d | vocab: %d'
		% (n_params / 1e6, config['model.args.base_type'], hidden, patch_size, vocab))

	paths = ensure_export(model, hidden, vocab, patch_size, out_dir, reexport=args.reexport)
	gen = MidiOrtKVGenerator(model, tokenizer, paths, threads=args.threads)

	if args.fidelity:
		print('\n===== FIDELITY (torch-fp32 vs int8 KV) =====')
		fidelity_check(model, gen, tokenizer)

	prompt = ''
	if args.prompt_file:
		with open(args.prompt_file) as f:
			prompt = f.read()

	print('\n===== GENERATE (max_patches=%d, temp=%.2f, top_k=%d, top_p=%.2f) ====='
		% (args.max_patches, args.temperature, args.top_k, args.top_p))
	t0 = time.perf_counter()
	text = gen.generate(prompt_text=prompt, max_patches=args.max_patches,
		temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, verbose=True)
	dt = time.perf_counter() - t0
	n_lines = text.count('\n') + 1 if text else 0
	print('\n[done] %.1fs  %d lines  %d chars' % (dt, n_lines, len(text)))

	# sanity: the generated MidiText should re-encode cleanly (no dropped/unknown events)
	reenc, dropped = tokenizer.encode_patches(text, add_special_patches=False)
	print('[check] re-encode: %d patches, %d dropped lines' % (len(reenc), len(dropped)))

	if args.output:
		with open(args.output, 'w') as f:
			f.write(text + ('\n' if not text.endswith('\n') else ''))
		print('[wrote]', args.output)


if __name__ == '__main__':
	main()
