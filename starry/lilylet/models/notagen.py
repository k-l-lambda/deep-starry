
'''
NotaGen-style hierarchical patch/token model for Lilylet, adapted to deep-starry conventions.

Reference: /home/camus/work/NotaGen/pretrain/utils.py

The model has two levels:
	- PatchLevelDecoder: encodes a sequence of patches (each patch is a fixed-length
	  sequence of token ids) into per-patch hidden states with a GPT2 backbone.
	- TokenLevelDecoder: an auto-regressive GPT2 LM that, conditioned on a patch's encoded
	  hidden state, generates the token ids inside that patch.

Inference model (deducer): LilyletNotaGen
Loss model (training wrapper): LilyletNotaGenLoss

Batch contract (from LilyletPatchy.collateBatch):
	input_patches: LongTensor [B, T, patch_size]   token ids in [0, token_vocab_size)
	input_masks:   LongTensor [B, T]               1 for real patch, 0 for padding
'''

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, PreTrainedModel
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers import PretrainedConfig


PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


def token_embedding_weight (base):
	'''Return the input token-embedding weight of a HF base model, regardless of
	architecture: GPT2 stores it at `transformer.wte`, Llama at `model.embed_tokens`.'''
	if hasattr(base, 'transformer'):			# GPT2Model / GPT2LMHeadModel
		return base.transformer.wte.weight
	if hasattr(base, 'model'):				# LlamaForCausalLM
		return base.model.embed_tokens.weight
	if hasattr(base, 'embed_tokens'):			# LlamaModel
		return base.embed_tokens.weight
	raise AttributeError(f'cannot locate token embedding on {type(base).__name__}')


class PatchLevelDecoder (PreTrainedModel):
	'''Encodes patches into per-patch hidden states (auto-regressive over patches).'''

	config_class = PretrainedConfig

	def __init__ (self, config, patch_size, token_vocab_size):
		super().__init__(config)

		self.patch_size = patch_size
		self.token_vocab_size = token_vocab_size

		hidden = getattr(config, 'n_embd', None) or config.hidden_size
		self.patch_embedding = nn.Linear(patch_size * token_vocab_size, hidden)
		nn.init.normal_(self.patch_embedding.weight, std=0.02)

		self.base = LlamaModel(config) if isinstance(config, LlamaConfig) else GPT2Model(config)

	def forward (self, patches: torch.Tensor, masks: Optional[torch.Tensor] = None):
		# patches: [B, T, patch_size] -> one-hot [B, T, patch_size, token_vocab_size]
		patches = F.one_hot(patches.long(), num_classes=self.token_vocab_size).to(self.patch_embedding.weight.dtype)
		patches = patches.reshape(len(patches), -1, self.patch_size * self.token_vocab_size)
		patches = self.patch_embedding(patches)

		if masks is None:
			return self.base(inputs_embeds=patches)
		return self.base(inputs_embeds=patches, attention_mask=masks)


class TokenLevelDecoder (PreTrainedModel):
	'''Generates the tokens within a patch, conditioned on the patch hidden state.'''

	config_class = PretrainedConfig

	def __init__ (self, config):
		super().__init__(config)

		self.special_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.base = LlamaForCausalLM(config) if isinstance(config, LlamaConfig) else GPT2LMHeadModel(config)

	def forward (self, encoded_patches: torch.Tensor, target_patches: torch.Tensor):
		# target_patches: [N, patch_size]; prepend BOS -> [N, patch_size + 1]
		target_patches = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.bos_token_id, target_patches), dim=1)

		# labels: mask padding positions with -100
		labels = target_patches.clone()
		labels = labels.masked_fill(labels == self.special_token_id, -100)

		# attention mask over tokens: 1 where label is valid, 0 where -100
		target_masks = torch.ones_like(labels)
		target_masks = target_masks.masked_fill(labels == -100, 0)

		# token embeddings, replace first position with the encoded patch state
		inputs_embeds = F.embedding(target_patches, token_embedding_weight(self.base))
		inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:, 1:, :]), dim=1)

		return self.base(inputs_embeds=inputs_embeds, attention_mask=target_masks, labels=labels)


class LilyletNotaGen (nn.Module):
	'''Inference model: hierarchical patch-level + token-level decoders.

	Args (from config['model.args']):
		token_vocab_size: token vocabulary size (Lilylet manual tokenizer = 256)
		patch_size: tokens per patch (16)
		patch_length: max patches per document (used for position embeddings)
		hidden_size: GPT2 embedding dim
		patch_num_layers: layers of the patch-level GPT2
		token_num_layers: layers of the token-level GPT2
		n_head: attention heads (default hidden_size // 64)
		base_type: 'gpt2' (default) or 'llama' backbone for both decoders
		intermediate_size: Llama FFN dim (default hidden_size * 4; ignored for gpt2)
		num_key_value_heads: Llama GQA kv heads (default n_head; ignored for gpt2)

	Backward compat: the legacy arg names `char_vocab_size` / `char_num_layers`
	are still accepted as aliases for `token_vocab_size` / `token_num_layers`, so
	configs (and .state.yaml) written before the rename keep loading.
	'''

	def __init__ (self, token_vocab_size=None, patch_size=16, patch_length=2048,
		hidden_size=768, patch_num_layers=12, token_num_layers=None, n_head=None,
		base_type='gpt2', intermediate_size=None, num_key_value_heads=None,
		char_vocab_size=None, char_num_layers=None, **_):
		super().__init__()

		# legacy aliases (char_* -> token_*) for pre-rename configs/checkpoints
		token_vocab_size = token_vocab_size if token_vocab_size is not None else (char_vocab_size if char_vocab_size is not None else 256)
		token_num_layers = token_num_layers if token_num_layers is not None else (char_num_layers if char_num_layers is not None else 3)

		self.token_vocab_size = token_vocab_size
		self.patch_size = patch_size
		self.base_type = base_type
		self.special_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.eos_token_id = EOS_TOKEN_ID

		n_head = n_head or max(1, hidden_size // 64)

		if base_type == 'llama':
			# Llama base: RoPE positions (no learned position table), GQA-capable.
			inter = intermediate_size or hidden_size * 4
			n_kv = num_key_value_heads or n_head
			patch_config = LlamaConfig(
				num_hidden_layers=patch_num_layers,
				max_position_embeddings=patch_length,
				hidden_size=hidden_size,
				intermediate_size=inter,
				num_attention_heads=n_head,
				num_key_value_heads=n_kv,
				vocab_size=1,
			)
			token_config = LlamaConfig(
				num_hidden_layers=token_num_layers,
				max_position_embeddings=patch_size + 1,
				hidden_size=hidden_size,
				intermediate_size=inter,
				num_attention_heads=n_head,
				num_key_value_heads=n_kv,
				vocab_size=token_vocab_size,
			)
		elif base_type == 'gpt2':
			patch_config = GPT2Config(
				num_hidden_layers=patch_num_layers,
				max_length=patch_length,
				max_position_embeddings=patch_length,
				n_embd=hidden_size,
				num_attention_heads=n_head,
				vocab_size=1,
			)
			token_config = GPT2Config(
				num_hidden_layers=token_num_layers,
				max_length=patch_size + 1,
				max_position_embeddings=patch_size + 1,
				hidden_size=hidden_size,
				num_attention_heads=n_head,
				vocab_size=token_vocab_size,
			)
		else:
			raise ValueError(f'Unknown base_type "{base_type}" (expected "gpt2" or "llama")')

		self.patch_level_decoder = PatchLevelDecoder(patch_config, patch_size, token_vocab_size)
		self.token_level_decoder = TokenLevelDecoder(token_config)

	def forward (self, patches: torch.Tensor, masks: torch.Tensor):
		'''
		patches: [B, T, patch_size] token ids
		masks:   [B, T] 1 for real patch, 0 for padding
		Returns (output, target_patches):
			output: token-level GPT2 output (has .loss and .logits) for next-patch prediction
			target_patches: [N, patch_size] the target tokens aligned to each prediction
		'''
		patches = patches.reshape(len(patches), -1, self.patch_size)
		encoded_patches = self.patch_level_decoder(patches, masks)['last_hidden_state']

		# Pair each encoded patch with the *next* patch's tokens (next-patch prediction):
		# - left_shift_masks selects encoded positions that have a valid following patch
		#   (i.e. drop the last real patch of each sequence).
		# - masks[:, 0] = 0 drops the first patch from the targets, so target i aligns to
		#   encoded position i-1.
		masks = masks.clone()
		left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
		masks[:, 0] = 0

		encoded_patches = encoded_patches[left_shift_masks == 1]
		target_patches = patches[masks == 1]

		return self.token_level_decoder(encoded_patches, target_patches), target_patches


class LilyletNotaGenLoss (nn.Module):
	'''Training wrapper: computes loss + metrics from a LilyletPatchy batch.'''

	def __init__ (self, **kw_args):
		super().__init__()

		self.deducer = LilyletNotaGen(**kw_args)

	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())

	def validation_parameters (self):
		return []

	def _token_accuracy (self, output, target_patches):
		# next-token accuracy over valid (non-pad) token positions, matching the
		# label shift the token-level GPT2 applies internally.
		token_targets = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.deducer.bos_token_id, target_patches), dim=1)
		labels = token_targets.masked_fill(token_targets == self.deducer.special_token_id, -100)
		shift_logits = output.logits[:, :-1, :]
		shift_labels = labels[:, 1:]
		valid = shift_labels != -100
		if not valid.any():
			return 0.0
		return (shift_logits.argmax(dim=-1)[valid] == shift_labels[valid]).float().mean().item()

	def forward (self, batch):
		output, target = self.deducer(batch['input_patches'], batch['input_masks'])

		with torch.no_grad():
			acc = self._token_accuracy(output, target)

		return output.loss, {'acc': acc}

	def inspectRun (self, batch):
		output, target = self.deducer(batch['input_patches'], batch['input_masks'])

		return {
			'loss': output.loss.item(),
			'acc': self._token_accuracy(output, target),
			'logits': output.logits,
			'target_patches': target,
			'n_patches': int(target.shape[0]),
		}


# ---- thin tensor-in/tensor-out wrappers around the two transformer forwards,
# used for ONNX export (the cheap one-hot / embedding-lookup / patch-state splice
# stay outside, done in numpy/torch by the caller). ----

class PatchNet (nn.Module):
	'''patch ids [1,T,patch_size] -> patch hidden states [1,T,hidden].'''
	def __init__ (self, model):
		super().__init__()
		self.dec = model.patch_level_decoder
		self.token_vocab_size = model.token_vocab_size
		self.patch_size = model.patch_size

	def forward (self, patches):
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
