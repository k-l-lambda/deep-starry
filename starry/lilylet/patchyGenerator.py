
'''
Autoregressive generator for the Lilylet NotaGen hierarchical patch/token model.

Wraps a trained `LilyletNotaGen` (see starry/lilylet/models/notagen.py) with the
hierarchical decoding loop adapted from NotaGen's inference/inference.py:

	- The patch-level decoder encodes the patch sequence generated so far into a
	  per-step hidden state.
	- The token-level decoder autoregressively samples the `patch_size` token ids
	  inside the next patch, seeded by that hidden state.
	- Generation stops on an EOS patch `[bos, eos, ...]` or a patch-count cap.

Token ids are decoded back to text via the tokenizer's `text_by_id` table (NOT
raw chr(), because Lilylet has protected multi-char tokens and ids > 127).
'''

import torch
import torch.nn.functional as F
import re

from ..utils.model_factory import loadModel
from .data.patchifier import LilyletTokenizer
from .models.notagen import token_embedding_weight


def sample_next (logits, temperature=1.0, top_k=0, top_p=1.0):
	'''Sample a single token id from a logits vector with temperature/top-k/top-p.'''
	logits = logits.float()
	if temperature != 1.0:
		logits = logits / max(temperature, 1e-6)
	if top_k and top_k > 0:
		k = min(top_k, logits.size(-1))
		kth = torch.topk(logits, k).values[..., -1, None]
		logits = logits.masked_fill(logits < kth, float('-inf'))
	if top_p and top_p < 1.0:
		sorted_logits, sorted_idx = torch.sort(logits, descending=True)
		probs = F.softmax(sorted_logits, dim=-1)
		cdf = probs.cumsum(dim=-1)
		remove = cdf > top_p
		remove[..., 1:] = remove[..., :-1].clone()
		remove[..., 0] = False
		sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
		logits = torch.full_like(logits, float('-inf')).scatter(-1, sorted_idx, sorted_logits)
	probs = F.softmax(logits, dim=-1)
	return int(torch.multinomial(probs, num_samples=1).item())


class LilyletPatchyGenerator:
	'''Loads a LilyletNotaGen checkpoint and runs autoregressive generation.'''

	def __init__ (self, model, tokenizer, device='cpu'):
		self.model = model.to(device).eval()
		self.tokenizer = tokenizer
		self.device = torch.device(device)

		self.patch_size = model.patch_size
		self.pad_id = model.special_token_id
		self.bos_id = model.bos_token_id
		self.eos_id = model.eos_token_id

	@classmethod
	def load (cls, checkpoint_path, tokenizer_path, model_args, device=None):
		'''Build LilyletNotaGen(**model_args), load weights, wrap in a generator.'''
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'

		model = loadModel({'type': 'LilyletNotaGen', 'args': model_args})
		checkpoint = torch.load(checkpoint_path, map_location='cpu')
		state = checkpoint['model'] if 'model' in checkpoint else checkpoint
		# legacy checkpoints store the token-level decoder as `char_level_decoder.*`;
		# remap to the renamed `token_level_decoder.*` so they still load.
		state = {
			(k.replace('char_level_decoder.', 'token_level_decoder.', 1) if k.startswith('char_level_decoder.') else k): v
			for k, v in state.items()
		}
		missing, unexpected = model.load_state_dict(state, strict=False)
		if missing or unexpected:
			print(f'[LilyletPatchyGenerator] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys')

		tokenizer = LilyletTokenizer(cls._resolve_tokenizer(tokenizer_path))
		return cls(model, tokenizer, device=device)

	@staticmethod
	def _resolve_tokenizer (tokenizer_path):
		'''Resolve a tokenizer path robustly: use it as-is if it exists, otherwise
		(for a repo-relative path like "assets/lilylet-tokenizer.json") resolve it
		against the repo root, so loading doesn't depend on the cwd.'''
		import os
		if os.path.isfile(tokenizer_path):
			return tokenizer_path
		if not os.path.isabs(tokenizer_path):
			# repo root = .../deep-starry, two levels up from this file's package dir
			repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
			candidate = os.path.join(repo_root, tokenizer_path)
			if os.path.isfile(candidate):
				return candidate
		return tokenizer_path

	@classmethod
	def from_config (cls, config, checkpoint_path, tokenizer_path=None, device=None):
		'''Build from a deep-starry Configuration: reads model.args and (unless
		overridden) data.args.tokenizer_path. Pass tokenizer_path explicitly when the
		config stores a repo-relative path but the cwd is elsewhere.'''
		model_args = dict(config['model.args'])
		if tokenizer_path is None:
			tokenizer_path = config['data.args.tokenizer_path']
		return cls.load(checkpoint_path, tokenizer_path, model_args, device=device)

	def patch_to_text (self, patch):
		'''Decode one patch (iterable of ids) to text: skip pad/bos, stop at eos.'''
		out = []
		for tid in patch:
			tid = int(tid)
			if tid == self.eos_id:
				break
			if tid in (self.pad_id, self.bos_id):
				continue
			out.append(self.tokenizer.text_by_id.get(tid, ''))
		return ''.join(out)

	def patches_to_text (self, patches):
		return ''.join(self.patch_to_text(p) for p in patches)

	@torch.no_grad()
	def generate_patch (self, encoded_patch, prefix_ids=None, temperature=1.0, top_k=0, top_p=1.0):
		'''Sample the token ids inside one patch, conditioned on a patch hidden state.

		encoded_patch: [hidden_size] hidden state from the patch-level decoder.
		prefix_ids: optional ids already fixed at the start of this patch.
		Returns a list of exactly patch_size ids.
		'''
		dec = self.model.token_level_decoder
		wte = token_embedding_weight(dec.base)
		# position 0 holds the encoded patch state; positions 1.. are embedded tokens.
		tokens = [self.bos_id] + list(prefix_ids or [])
		generated = list(prefix_ids or [])
		encoded = encoded_patch.reshape(1, 1, -1)
		while len(generated) < self.patch_size:
			tok_tensor = torch.tensor([tokens], device=self.device)
			emb = F.embedding(tok_tensor, wte)
			emb = torch.cat((encoded, emb[:, 1:, :]), dim=1)
			logits = dec.base(inputs_embeds=emb).logits[0, -1]
			nxt = sample_next(logits, temperature=temperature, top_k=top_k, top_p=top_p)
			generated.append(nxt)
			tokens.append(nxt)
		return generated

	@torch.no_grad()
	def generate (self, prompt_text='', max_patches=256, temperature=1.0, top_k=0, top_p=0.9,
		measures=None, postprocess=False, verbose=False):
		'''Autoregressively generate a Lilylet document.

		Seeds with a BOS patch (+ optional metadata prompt), then samples patch by
		patch until an EOS patch appears or max_patches is reached. Returns the
		decoded text.

		measures: if set, forces the first body patch to start with `[r:0/<measures>`
			(a one-shot priming of the stream marker, like NotaGen's `[r:0/`); if None,
			the model generates the marker on its own.
		postprocess: if True, run self.postprocess on the result (drop `[r:x/y]`
			markers, insert blank lines after the meta block and at measure boundaries).
		'''
		bos_patch = [self.bos_id] * (self.patch_size - 1) + [self.eos_id]
		patches = [bos_patch]

		if prompt_text:
			for line in prompt_text.splitlines():
				ids = self.tokenizer.encode(line + '\n')
				for i in range(0, len(ids), self.patch_size):
					chunk = ids[i:i + self.patch_size]
					patches.append(chunk + [self.pad_id] * (self.patch_size - len(chunk)))

		out_text = self.patches_to_text(patches[1:])
		if verbose and out_text:
			print(out_text, end='')

		# one-shot priming prefix for the first body (stream) patch. Includes the
		# closing `]` so the remaining-measure count is exactly `measures - 1` and the
		# model can't extend the digits (e.g. 8 -> 81). The marker is 0-based and `y`
		# counts measures remaining AFTER this one (see patchifier: y = total - i - 1),
		# so `[r:0/{measures-1}]` yields exactly `measures` total measures.
		prime_ids = self.tokenizer.encode(f'[r:0/{measures - 1}]') if measures is not None and measures >= 1 else None
		primed = False

		for _ in range(max_patches):
			inp = torch.tensor([sum(patches, [])], device=self.device).reshape(1, -1, self.patch_size)
			encoded = self.model.patch_level_decoder(inp)['last_hidden_state']
			last = encoded[0, -1]

			patch_ids = self.generate_patch(last, temperature=temperature, top_k=top_k, top_p=top_p)

			# first time the model emits a stream patch, re-sample it with the forced
			# `[r:0/<measures>` prefix so the body starts at the requested measure count.
			if prime_ids is not None and not primed and self.patch_to_text(patch_ids).startswith('[r:'):
				primed = True
				patch_ids = self.generate_patch(last, prefix_ids=prime_ids,
					temperature=temperature, top_k=top_k, top_p=top_p)

			# EOS patch -> done
			if patch_ids[0] == self.bos_id and patch_ids[1] == self.eos_id:
				if verbose:
					print('\n[EOS patch -> stop]')
				break

			text = self.patch_to_text(patch_ids)
			out_text += text
			if verbose:
				print(text, end='')

			# mask tokens after the first EOS inside the patch to PAD before appending
			clean = list(patch_ids)
			seen_eos = False
			for j in range(len(clean)):
				if seen_eos:
					clean[j] = self.pad_id
				if clean[j] == self.eos_id:
					seen_eos = True
			patches.append(clean)
		else:
			if verbose:
				print('\n[max_patches reached]')

		return self.postprocess(out_text) if postprocess else out_text

	# matches a stream marker like `[r:12/34]` (and a trailing partial `[r:12/` at EOF);
	# group 1 captures the `x/y` payload.
	_STREAM_RE = re.compile(r'\[r:(\d+/\d*)\]?')

	def postprocess (self, text):
		'''Clean generated Lilylet for readability:
		- move each `[r:x/y]` stream marker from the measure head to a trailing
		  comment `% r:x/y` at the measure end (the line ending in `|`),
		- insert a blank line after the metadata block,
		- insert a blank line at every measure boundary.
		'''
		lines = [ln.rstrip() for ln in text.split('\n')]
		out = []
		meta_done = False
		pending = None		# marker payload waiting to be attached at the measure end

		for ln in lines:
			# extract the stream marker (sits at the measure head); remember its
			# payload and strip the marker text from the line.
			m = self._STREAM_RE.search(ln)
			if m:
				# a still-pending marker means the previous measure had no barline
				# (e.g. truncated); keep it as a standalone comment so it isn't lost.
				if pending is not None:
					out.append('% r:' + pending)
				pending = m.group(1)
				ln = self._STREAM_RE.sub('', ln).rstrip()

			# metadata lines: `[field "..."]` headers and leading `%<style>` comments
			# (the --styles-in-comments format). Both belong to the meta block, so the
			# blank-line separator goes after the last of them, not before a `%` style line.
			is_meta = (ln.startswith('[') and ln.endswith(']')) or (ln.startswith('%') and not ln.startswith('%%'))
			# blank line once the metadata block ends and the body begins
			if not meta_done and out and not is_meta and ln:
				out.append('')
				meta_done = True

			is_measure_end = ln.endswith('|')
			if is_measure_end and pending is not None:
				ln = ln + ' % r:' + pending
				pending = None

			out.append(ln)

			# blank line after a measure-ending line
			if meta_done and is_measure_end:
				out.append('')

		# any marker left pending at EOF -> attach as a trailing comment
		if pending is not None:
			out.append('% r:' + pending)

		# collapse runs of blank lines, trim leading/trailing blanks
		cleaned = []
		for ln in out:
			if ln == '' and (not cleaned or cleaned[-1] == ''):
				continue
			cleaned.append(ln)
		return '\n'.join(cleaned).strip('\n') + '\n'
