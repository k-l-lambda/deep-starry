
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

from transformers import GPT2Config, LlamaConfig
from ...utils.registry import register_model

# Generic bGPT two-level decoder building blocks now live in starry.bgpt. They are
# re-exported here so existing references (patchyGenerator, the ORT export tool, the
# benchmarks) keep importing them from this module unchanged.
from ...bgpt.decoders import (
	PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID,
	token_embedding_weight,
	PatchLevelDecoder, TokenLevelDecoder,
)
from ...bgpt.kv_net import PatchNet, TokenNet, PatchNetKV, TokenNetKV


@register_model
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

	def forward (self, patches: torch.Tensor, masks: torch.Tensor, target_masks: Optional[torch.Tensor] = None):
		'''
		patches: LongTensor [B, T, patch_size] token ids
		masks:   LongTensor [B, T] 1 for real patch, 0 for padding (the patch-level ATTENTION
		         mask; prompt patches stay 1 here so the model conditions on them).
		target_masks: LongTensor [B, T] or None. 1 where the patch is a supervised prediction
		         TARGET, 0 over the prompt + <bos> boundary + padding. When None, falls back to
		         the legacy behavior (supervise every real patch except the first).
		Returns (output, target_patches), where N = number of target patches across the batch:
			output: token-level decoder output (has .loss scalar and
			        .logits [N, patch_size + 1, token_vocab_size]) for next-patch prediction
			target_patches: LongTensor [N, patch_size] the target tokens aligned to each prediction
		'''
		patches = patches.reshape(len(patches), -1, self.patch_size)
		encoded_patches = self.patch_level_decoder(patches, masks)['last_hidden_state']

		# Next-patch prediction: encoded patch i predicts patch i+1. A patch is a target iff
		# target_masks==1; an encoded position is a supervised INPUT iff its next patch is a
		# target. So left_shift_masks = target_masks shifted left by one, gated by the
		# attention mask (the input itself must be a real patch).
		if target_masks is None:
			target_masks = masks.clone()
			target_masks[:, 0] = 0  # legacy: drop the first (<bos>) patch from targets
		else:
			target_masks = target_masks.clone()

		left_shift_masks = torch.zeros_like(masks)
		left_shift_masks[:, :-1] = target_masks[:, 1:]
		left_shift_masks = left_shift_masks * masks

		encoded_patches = encoded_patches[left_shift_masks == 1]
		target_patches = patches[target_masks == 1]

		return self.token_level_decoder(encoded_patches, target_patches), target_patches


@register_model
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
		'''Next-token accuracy over valid (non-pad) token positions, reproducing the
		label shift the token-level decoder applies internally.

		output:         the token-level decoder output; .logits is [N, patch_size + 1, token_vocab_size]
		target_patches: LongTensor [N, patch_size] the target tokens (N = #target patches)
		Returns: float scalar accuracy (0.0 when there are no valid positions).
		'''
		# rebuild the same labels the decoder used: prepend <bos> -> [N, patch_size + 1],
		# then mask pad positions with -100 so they are excluded.
		token_targets = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.deducer.bos_token_id, target_patches), dim=1)
		labels = token_targets.masked_fill(token_targets == self.deducer.special_token_id, -100)
		# causal shift: logits at position i predict the token at position i+1.
		shift_logits = output.logits[:, :-1, :]		# [N, patch_size, token_vocab_size]
		shift_labels = labels[:, 1:]				# [N, patch_size]
		valid = shift_labels != -100
		if not valid.any():
			return 0.0
		return (shift_logits.argmax(dim=-1)[valid] == shift_labels[valid]).float().mean().item()

	def forward (self, batch):
		'''
		batch: dict from LilyletPatchy.collateBatch with
			input_patches LongTensor [B, T, patch_size]
			input_masks   LongTensor [B, T]            patch-level attention mask
			input_targets LongTensor [B, T] (optional) supervision mask; absent -> legacy behavior
		Returns (loss, metrics):
			loss:    scalar token-level cross-entropy
			metrics: {'acc': float next-token accuracy}
		'''
		output, target = self.deducer(batch['input_patches'], batch['input_masks'], batch.get('input_targets'))

		with torch.no_grad():
			acc = self._token_accuracy(output, target)

		return output.loss, {'acc': acc}

	def inspectRun (self, batch):
		output, target = self.deducer(batch['input_patches'], batch['input_masks'], batch.get('input_targets'))

		return {
			'loss': output.loss.item(),
			'acc': self._token_accuracy(output, target),
			'logits': output.logits,
			'target_patches': target,
			'n_patches': int(target.shape[0]),
		}
