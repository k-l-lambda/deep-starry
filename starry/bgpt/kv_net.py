'''
ONNX-export wrappers for the generic bGPT two-level decoders.

Thin tensor-in/tensor-out modules around the PatchLevelDecoder / TokenLevelDecoder
forwards (see starry.bgpt.decoders), used for ONNX export. They reference only the
generic attributes a bGPT deducer exposes — `model.patch_level_decoder`,
`model.token_level_decoder`, `model.token_vocab_size`, `model.patch_size` — so any
bGPT model (Lilylet, MIDI, …) can export through them.

The cheap one-hot / embedding-lookup / patch-state splice stay outside the graph
(done in numpy/torch by the caller); the KV variants keep only the per-layer K/V
tensors between steps for O(T) incremental decoding.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNet (nn.Module):
	'''patch ids [1,T,patch_size] -> patch hidden states [1,T,hidden].'''
	def __init__ (self, model):
		super().__init__()
		self.dec = model.patch_level_decoder
		self.token_vocab_size = model.token_vocab_size
		self.patch_size = model.patch_size

	def forward (self, patches):
		'''patches: LongTensor [1, T, patch_size] -> last_hidden FloatTensor [1, T, hidden].'''
		oh = F.one_hot(patches.long(), num_classes=self.token_vocab_size).to(self.dec.patch_embedding.weight.dtype)
		oh = oh.reshape(1, -1, self.patch_size * self.token_vocab_size)
		emb = self.dec.patch_embedding(oh)
		return self.dec.base(inputs_embeds=emb).last_hidden_state


class TokenNet (nn.Module):
	'''token inputs_embeds [1,L,hidden] -> logits [1,L,vocab]. Embedding lookup +
	the position-0 patch-state splice stay outside (cheap, done in numpy/torch).'''
	def __init__ (self, model):
		super().__init__()
		self.base = model.token_level_decoder.base

	def forward (self, inputs_embeds):
		'''inputs_embeds: FloatTensor [1, L, hidden] -> logits FloatTensor [1, L, token_vocab_size].'''
		return self.base(inputs_embeds=inputs_embeds).logits


class PatchNetKV (nn.Module):
	'''KV-cache variant of PatchNet for incremental patch-level decoding.

	Input:  patches [1, L, patch_size]  (the L new patches, usually L=1 in the loop,
	            L>1 only for the initial prefill of the seed patches)
	        past:   list of 2*num_layers tensors  [k0, v0, k1, v1, ...], each
	            [1, num_kv_heads, P, head_dim] (P = cached patch length, 0 at prefill)
	Output: (last_hidden [1, L, hidden], new_k0, new_v0, new_k1, new_v1, ...)
	        where each new_k/new_v is [1, num_kv_heads, P+L, head_dim].

	The one-hot + patch_embedding stay inside the graph (so they get quantized);
	the caller keeps only the per-layer K/V tensors between steps.
	'''
	def __init__ (self, model):
		super().__init__()
		self.dec = model.patch_level_decoder
		self.token_vocab_size = model.token_vocab_size
		self.patch_size = model.patch_size
		self.num_layers = self.dec.base.config.num_hidden_layers

	def forward (self, patches, past):
		'''
		patches: LongTensor [1, L, patch_size]   the L new patches (L=1 in the loop)
		past:    list of 2*num_layers tensors [k0, v0, ...], each [1, num_kv_heads, P, head_dim]
		Returns: tuple (last_hidden [1, L, hidden], new_k0, new_v0, ...) with each
		         new_k/new_v of shape [1, num_kv_heads, P+L, head_dim].
		'''
		from transformers import DynamicCache
		oh = F.one_hot(patches.long(), num_classes=self.token_vocab_size).to(self.dec.patch_embedding.weight.dtype)
		oh = oh.reshape(1, -1, self.patch_size * self.token_vocab_size)
		emb = self.dec.patch_embedding(oh)

		cache = DynamicCache()
		past_len = past[0].shape[2]
		for i in range(self.num_layers):
			cache.update(past[2 * i], past[2 * i + 1], i)

		cache_position = torch.arange(past_len, past_len + emb.shape[1])
		out = self.dec.base(inputs_embeds=emb, past_key_values=cache, use_cache=True,
			cache_position=cache_position)

		outs = [out.last_hidden_state]
		for i in range(self.num_layers):
			outs.append(cache.layers[i].keys)
			outs.append(cache.layers[i].values)
		return tuple(outs)


class TokenNetKV (nn.Module):
	'''KV-cache variant of TokenNet for incremental token-level decoding.

	Input:  inputs_embeds [1, L, hidden]  (the L new token embeddings; L=1 in the
	            loop, the first step's position 0 holds the patch hidden state)
	        past:   list of 2*num_layers tensors [k0, v0, k1, v1, ...], each
	            [1, num_kv_heads, P, head_dim] (P = cached token length, 0 at prefill)
	Output: (logits [1, L, vocab], new_k0, new_v0, ...)  each new_k/v [1,NKV,P+L,HD].

	The embedding lookup + the position-0 patch-state splice stay outside (done in
	numpy/torch by the caller), exactly like TokenNet.
	'''
	def __init__ (self, model):
		super().__init__()
		self.base = model.token_level_decoder.base
		self.num_layers = self.base.config.num_hidden_layers

	def forward (self, inputs_embeds, past):
		'''
		inputs_embeds: FloatTensor [1, L, hidden]   the L new token embeddings (L=1 in the loop)
		past:          list of 2*num_layers tensors [k0, v0, ...], each [1, num_kv_heads, P, head_dim]
		Returns: tuple (logits [1, L, token_vocab_size], new_k0, new_v0, ...) with each
		         new_k/new_v of shape [1, num_kv_heads, P+L, head_dim].
		'''
		from transformers import DynamicCache
		cache = DynamicCache()
		past_len = past[0].shape[2]
		for i in range(self.num_layers):
			cache.update(past[2 * i], past[2 * i + 1], i)

		cache_position = torch.arange(past_len, past_len + inputs_embeds.shape[1])
		out = self.base(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True,
			cache_position=cache_position)

		outs = [out.logits]
		for i in range(self.num_layers):
			outs.append(cache.layers[i].keys)
			outs.append(cache.layers[i].values)
		return tuple(outs)
