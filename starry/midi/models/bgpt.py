'''
bGPT-style hierarchical patch/token generation model for MIDI-text, adapted to
deep-starry conventions.

Architecture (shared, modality-agnostic — see starry.bgpt.decoders):
	- PatchLevelDecoder: encodes a sequence of event-patches (each patch = the fixed-length
	  token ids of one MidiText event) into per-patch hidden states with a GPT2/Llama backbone.
	- TokenLevelDecoder: an auto-regressive LM that, conditioned on a patch's encoded hidden
	  state, generates the token ids inside that patch.

This is the MIDI counterpart of starry.lilylet.models.notagen.LilyletNotaGen. The two
share the exact same two-level bGPT core and the same batch contract; they differ only in
modality defaults (token_vocab_size, the MidiTokenizer vocab = 37) and naming.

Inference model (deducer): MidiBGPT
Loss model (training wrapper): MidiBGPTLoss

Batch contract (from MidiPatchy.collateBatch, identical to LilyletPatchy):
	input_patches: LongTensor [B, T, patch_size]   token ids in [0, token_vocab_size)
	input_masks:   LongTensor [B, T]               1 for real patch, 0 for padding
	input_targets: LongTensor [B, T]               1 where the patch is a supervised target
'''

from typing import Optional
import torch
import torch.nn as nn

from transformers import GPT2Config, LlamaConfig
from ...utils.registry import register_model

# Generic bGPT two-level decoder building blocks live in starry.bgpt and are re-exported
# here so downstream references (an ORT export tool, benchmarks) can import them from this
# module unchanged, exactly as starry.lilylet.models.notagen does.
from ...bgpt.decoders import (
	PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID,
	token_embedding_weight,
	PatchLevelDecoder, TokenLevelDecoder,
)
from ...bgpt.kv_net import PatchNet, TokenNet, PatchNetKV, TokenNetKV

# Within an event-patch the token layout is [event_token, deltaTime hex digits..., space,
# fields...], so the deltaTime occupies positions 1 .. (first space - 1). The space token
# id is looked up lazily (and cached) from the MidiTokenizer vocab so it tracks vocab
# changes rather than hard-coding it.
_SPACE_TOKEN_ID = None

def _space_token_id ():
	global _SPACE_TOKEN_ID
	if _SPACE_TOKEN_ID is None:
		from ..tokenizer import MidiTokenizer
		_SPACE_TOKEN_ID = MidiTokenizer().id_by_token[' ']
	return _SPACE_TOKEN_ID


@register_model
class MidiBGPT (nn.Module):
	'''Inference model: hierarchical patch-level + token-level bGPT decoders for MIDI-text.

	Args (from config['model.args']):
		token_vocab_size: token vocabulary size (MidiTokenizer = 37)
		patch_size: tokens per event-patch (16)
		patch_length: max patches per document (used for position embeddings)
		hidden_size: GPT2/Llama embedding dim
		patch_num_layers: layers of the patch-level decoder
		token_num_layers: layers of the token-level decoder
		n_head: attention heads (default hidden_size // 64)
		base_type: 'gpt2' (default) or 'llama' backbone for both decoders
		intermediate_size: Llama FFN dim (default hidden_size * 4; ignored for gpt2)
		num_key_value_heads: Llama GQA kv heads (default n_head; ignored for gpt2)
	'''

	def __init__ (self, token_vocab_size=37, patch_size=16, patch_length=2048,
		hidden_size=768, patch_num_layers=12, token_num_layers=3, n_head=None,
		base_type='gpt2', intermediate_size=None, num_key_value_heads=None, **_):
		super().__init__()

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

	def forward (self, patches: torch.Tensor, masks: torch.Tensor, target_masks: Optional[torch.Tensor] = None,
		positions: Optional[torch.Tensor] = None):
		'''
		patches: LongTensor [B, T, patch_size] token ids
		masks:   LongTensor [B, T] 1 for real patch, 0 for padding (the patch-level ATTENTION
		         mask; prompt patches stay 1 here so the model conditions on them).
		positions: LongTensor [B, T] or None. Absolute patch indices in the original song;
		         when present, these are passed as patch-level position_ids so random-cropped
		         continuation windows keep their original position instead of restarting at 0.
		Returns (output, target_patches), where N = number of target patches across the batch:
			output: token-level decoder output (has .loss scalar and
			        .logits [N, patch_size + 1, token_vocab_size]) for next-patch prediction
			target_patches: LongTensor [N, patch_size] the target tokens aligned to each prediction
		'''
		patches = patches.reshape(len(patches), -1, self.patch_size)
		encoded_patches = self.patch_level_decoder(patches, masks, position_ids=positions)['last_hidden_state']

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
class MidiBGPTLoss (nn.Module):
	'''Training wrapper: computes loss + metrics from a MidiPatchy batch.'''

	def __init__ (self, **kw_args):
		super().__init__()

		self.deducer = MidiBGPT(**kw_args)

	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())

	def validation_parameters (self):
		return []

	def _token_accuracy (self, output, target_patches):
		'''Next-token accuracy over valid (non-pad) token positions, reproducing the label
		shift the token-level decoder applies internally.

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

	def _time_err (self, output, target_patches):
		'''Error rate over the deltaTime tokens only.

		Within each event-patch the layout is [event_token, deltaTime hex digits..., space,
		fields...]; target_patches is [N, patch_size] (position 0 = event token, position 1 =
		first deltaTime digit). The deltaTime occupies positions 1 .. (first space - 1). This
		reports the next-token error rate (1 - accuracy) restricted to those positions, i.e.
		how well the model predicts the inter-event timing. Returns 0.0 when there are no
		deltaTime positions in the batch.

		Computed on the same shifted alignment as _token_accuracy: in shift space, column k
		predicts target_patches[:, k].
		'''
		space_id = _space_token_id()
		N, P = target_patches.shape

		# per-patch deltaTime mask over target positions 0..P-1: position >= 1, and strictly
		# before that patch's first space token (the deltaTime/field separator).
		pos = torch.arange(P, device=target_patches.device).unsqueeze(0).expand(N, P)
		is_space = target_patches == space_id
		# index of the first space per patch; patches with no space (shouldn't happen for a
		# real event with fields) get P so the whole tail counts as deltaTime.
		has_space = is_space.any(dim=1)
		first_space = torch.where(
			has_space,
			is_space.float().argmax(dim=1),
			torch.full((N,), P, device=target_patches.device, dtype=torch.long),
		).unsqueeze(1)
		delta_mask = (pos >= 1) & (pos < first_space)
		# exclude any pad positions (deltaTime never pads, but be safe)
		delta_mask &= target_patches != self.deducer.special_token_id

		if not delta_mask.any():
			return 0.0
		# shift space: shift_logits[:, k] predicts target_patches[:, k]
		shift_logits = output.logits[:, :-1, :]		# [N, P, vocab]
		preds = shift_logits.argmax(dim=-1)			# [N, P]
		sel = delta_mask
		acc = (preds[sel] == target_patches[sel]).float().mean().item()
		return 1.0 - acc

	def forward (self, batch):
		'''
		batch: dict from MidiPatchy.collateBatch with
			input_patches LongTensor [B, T, patch_size]
			input_masks   LongTensor [B, T]            patch-level attention mask
			input_targets LongTensor [B, T] (optional) supervision mask; absent -> legacy behavior
		Returns (loss, metrics):
			loss:    scalar token-level cross-entropy
			metrics: {'acc': float next-token accuracy, 'err': 1 - acc}
			         plus 'time_err' (deltaTime-only error rate) ONLY in eval mode — the
			         trainer runs validation under model.eval(), so time_err is computed on
			         the val set only and never slows the training step.
		'''
		output, target = self.deducer(
			batch['input_patches'], batch['input_masks'], batch.get('input_targets'), batch.get('input_positions'))

		with torch.no_grad():
			acc = self._token_accuracy(output, target)
			metrics = {'acc': acc, 'err': 1 - acc}
			if not self.training:
				metrics['time_err'] = self._time_err(output, target)

		return output.loss, metrics

	def inspectRun (self, batch):
		output, target = self.deducer(
			batch['input_patches'], batch['input_masks'], batch.get('input_targets'), batch.get('input_positions'))

		acc = self._token_accuracy(output, target)
		return {
			'loss': output.loss.item(),
			'acc': acc,
			'err': 1 - acc,
			'time_err': self._time_err(output, target),
			'logits': output.logits,
			'target_patches': target,
			'n_patches': int(target.shape[0]),
		}
