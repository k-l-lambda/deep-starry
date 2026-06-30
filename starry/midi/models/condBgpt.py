'''Conditioned-MIDI bGPT: generate MIDI event patches conditioned on a lilylet score.

A joint-sequence model over [lilylet score patches] ++ [midi event patches]. The two
modalities keep their OWN token ids and get SEPARATE patch-embedding modules, but share one
patch-level Llama backbone; a custom 4D attention mask (built by CondMidiPatchy.collateBatch)
couples midi measures to lilylet measures. Only the midi segment is supervised — the lilylet
segment is a read-only condition — so there is a SINGLE token-level head (midi vocab).

Inference model (deducer): CondMidiBGPT
Loss model (training wrapper): CondMidiBGPTLoss

Batch contract (from starry.midi.data.condPatchy.CondMidiPatchy.collateBatch):
	input_patches   LongTensor [B, T, patch_size]
	input_masks     LongTensor [B, T]               1 = real patch, 0 = padding
	input_targets   LongTensor [B, T]               1 = supervised midi target
	input_positions LongTensor [B, T]
	attn_mask       BoolTensor [B, 1, T, T]          custom windowed/block mask
	modality        LongTensor [B, T]               0 = lilylet, 1 = midi
	lyl_counts      LongTensor [B]
'''

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, PreTrainedModel, PretrainedConfig
from ...utils.registry import register_model
from ...bgpt.decoders import (
	PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID,
	token_embedding_weight, TokenLevelDecoder,
)
from .bgpt import _space_token_id


class DualPatchLevelDecoder (PreTrainedModel):
	'''Patch-level decoder with TWO patch embeddings (lilylet / midi) + one shared backbone.

	Patches in the lilylet segment (positions < lyl_count) are embedded by lyl_embed (vocab
	256); midi-segment patches by midi_embed (vocab 41). Both project to the same hidden size,
	are concatenated back in sequence order, and run through one Llama backbone under the
	custom 4D attention mask.
	'''

	config_class = PretrainedConfig

	def __init__ (self, config, patch_size, lyl_vocab_size, midi_vocab_size):
		super().__init__(config)
		self.patch_size = patch_size
		self.lyl_vocab_size = lyl_vocab_size
		self.midi_vocab_size = midi_vocab_size
		hidden = getattr(config, 'n_embd', None) or config.hidden_size
		self.lyl_embedding = nn.Linear(patch_size * lyl_vocab_size, hidden)
		self.midi_embedding = nn.Linear(patch_size * midi_vocab_size, hidden)
		nn.init.normal_(self.lyl_embedding.weight, std=0.02)
		nn.init.normal_(self.midi_embedding.weight, std=0.02)
		self.base = LlamaModel(config)

	def forward (self, patches, modality, attn_mask=None, position_ids=None):
		'''
		patches:   LongTensor [B, T, patch_size]  token ids (lilylet ids then midi ids)
		modality:  LongTensor [B, T]              0 = lilylet, 1 = midi
		attn_mask: BoolTensor [B, 1, T, T] or None  custom mask (True = attend); passed as-is
		position_ids: LongTensor [B, T] or None
		Returns the HF base output; .last_hidden_state is [B, T, hidden].
		'''
		dtype = self.lyl_embedding.weight.dtype
		# Each one-hot is over its OWN vocab; clamp first so out-of-range ids of the WRONG
		# modality don't trip one_hot (those rows are discarded by the where-select anyway).
		lyl_ids = patches.clamp(max=self.lyl_vocab_size - 1)
		midi_ids = patches.clamp(max=self.midi_vocab_size - 1)
		oh_lyl = F.one_hot(lyl_ids.long(), self.lyl_vocab_size).to(dtype)
		oh_midi = F.one_hot(midi_ids.long(), self.midi_vocab_size).to(dtype)
		e_lyl = self.lyl_embedding(oh_lyl.reshape(len(patches), -1, self.patch_size * self.lyl_vocab_size))
		e_midi = self.midi_embedding(oh_midi.reshape(len(patches), -1, self.patch_size * self.midi_vocab_size))
		is_lyl = (modality == 0).unsqueeze(-1)			# [B,T,1]
		embeds = torch.where(is_lyl, e_lyl, e_midi)		# [B,T,hidden]

		if attn_mask is None:
			return self.base(inputs_embeds=embeds, position_ids=position_ids)
		return self.base(inputs_embeds=embeds, attention_mask=attn_mask, position_ids=position_ids)


@register_model
class CondMidiBGPT (nn.Module):
	'''Inference model: dual-embedding patch decoder + single midi token-level head.

	Args (from config['model.args']):
		lyl_vocab_size:  lilylet tokenizer vocab (256)
		midi_vocab_size: midi tokenizer vocab (41)
		patch_size: tokens per patch (16)
		patch_length: max patches per sequence (position embedding bound)
		hidden_size, patch_num_layers, token_num_layers, n_head, intermediate_size,
		num_key_value_heads: Llama backbone dims (llama base only)
	'''

	def __init__ (self, lyl_vocab_size=256, midi_vocab_size=41, patch_size=16, patch_length=4096,
		hidden_size=768, patch_num_layers=12, token_num_layers=3, n_head=None,
		base_type='llama', intermediate_size=None, num_key_value_heads=None, **_):
		super().__init__()
		assert base_type == 'llama', 'CondMidiBGPT supports the llama backbone only'
		self.lyl_vocab_size = lyl_vocab_size
		self.midi_vocab_size = midi_vocab_size
		self.patch_size = patch_size
		self.special_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.eos_token_id = EOS_TOKEN_ID

		n_head = n_head or max(1, hidden_size // 64)
		inter = intermediate_size or hidden_size * 4
		n_kv = num_key_value_heads or n_head

		patch_config = LlamaConfig(
			num_hidden_layers=patch_num_layers, max_position_embeddings=patch_length,
			hidden_size=hidden_size, intermediate_size=inter,
			num_attention_heads=n_head, num_key_value_heads=n_kv, vocab_size=1,
		)
		# token-level head decodes midi patches only -> midi vocab.
		token_config = LlamaConfig(
			num_hidden_layers=token_num_layers, max_position_embeddings=patch_size + 1,
			hidden_size=hidden_size, intermediate_size=inter,
			num_attention_heads=n_head, num_key_value_heads=n_kv, vocab_size=midi_vocab_size,
		)

		self.patch_level_decoder = DualPatchLevelDecoder(patch_config, patch_size, lyl_vocab_size, midi_vocab_size)
		self.token_level_decoder = TokenLevelDecoder(token_config)

	def forward (self, patches, masks, modality, attn_mask, target_masks=None,
		positions=None, lyl_counts=None):
		'''
		patches:      LongTensor [B, T, patch_size]
		masks:        LongTensor [B, T]            1 = real patch, 0 = padding
		modality:     LongTensor [B, T]            0 = lilylet, 1 = midi
		attn_mask:    BoolTensor [B, 1, T, T]      custom windowed/block mask
		target_masks: LongTensor [B, T]            1 = supervised midi target
		positions:    LongTensor [B, T] or None    patch-level position_ids
		Returns (token_output, target_patches): token-level next-patch prediction over the
		selected midi target patches (N = #targets across batch).
		'''
		patches = patches.reshape(len(patches), -1, self.patch_size)
		encoded = self.patch_level_decoder(patches, modality, attn_mask, position_ids=positions)['last_hidden_state']

		if target_masks is None:
			# fallback: supervise every real midi patch except the first patch overall.
			target_masks = (modality == 1).long() * masks
			target_masks[:, 0] = 0
		else:
			target_masks = target_masks.clone()

		# Next-patch prediction: encoded patch i predicts patch i+1. An encoded position is a
		# supervised INPUT iff its NEXT patch is a target; gate by the real-patch mask.
		left_shift = torch.zeros_like(masks)
		left_shift[:, :-1] = target_masks[:, 1:]
		left_shift = left_shift * masks

		encoded_sel = encoded[left_shift == 1]			# [N, hidden]
		target_patches = patches[target_masks == 1]		# [N, patch_size]
		return self.token_level_decoder(encoded_sel, target_patches), target_patches


@register_model
class CondMidiBGPTLoss (nn.Module):
	'''Training wrapper: loss + metrics from a CondMidiPatchy batch (midi-segment only).'''

	def __init__ (self, **kw_args):
		super().__init__()
		self.deducer = CondMidiBGPT(**kw_args)

	def training_parameters (self):
		return list(self.deducer.parameters()) + list(self.deducer.buffers())

	def validation_parameters (self):
		return []

	def _token_accuracy (self, output, target_patches):
		'''Next-token accuracy over valid (non-pad) midi token positions (mirrors the decoder's
		internal <bos>-prepend + -100 pad-mask + causal shift).'''
		token_targets = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.deducer.bos_token_id, target_patches), dim=1)
		labels = token_targets.masked_fill(token_targets == self.deducer.special_token_id, -100)
		shift_logits = output.logits[:, :-1, :]
		shift_labels = labels[:, 1:]
		valid = shift_labels != -100
		if not valid.any():
			return 0.0
		return (shift_logits.argmax(dim=-1)[valid] == shift_labels[valid]).float().mean().item()

	def _time_err (self, output, target_patches):
		'''Error rate over deltaTime tokens only (positions 1 .. first-space-1 of each patch).'''
		space_id = _space_token_id()
		N, P = target_patches.shape
		pos = torch.arange(P, device=target_patches.device).unsqueeze(0).expand(N, P)
		is_space = target_patches == space_id
		has_space = is_space.any(dim=1)
		first_space = torch.where(
			has_space, is_space.float().argmax(dim=1),
			torch.full((N,), P, device=target_patches.device, dtype=torch.long),
		).unsqueeze(1)
		delta_mask = (pos >= 1) & (pos < first_space)
		delta_mask &= target_patches != self.deducer.special_token_id
		if not delta_mask.any():
			return 0.0
		preds = output.logits[:, :-1, :].argmax(dim=-1)
		return 1.0 - (preds[delta_mask] == target_patches[delta_mask]).float().mean().item()

	def forward (self, batch):
		output, target = self.deducer(
			batch['input_patches'], batch['input_masks'], batch['modality'], batch['attn_mask'],
			batch.get('input_targets'), batch.get('input_positions'), batch.get('lyl_counts'))

		with torch.no_grad():
			acc = self._token_accuracy(output, target)
			metrics = {'acc': acc, 'err': 1 - acc}
			if not self.training:
				metrics['time_err'] = self._time_err(output, target)

		return output.loss, metrics

	def inspectRun (self, batch):
		output, target = self.deducer(
			batch['input_patches'], batch['input_masks'], batch['modality'], batch['attn_mask'],
			batch.get('input_targets'), batch.get('input_positions'), batch.get('lyl_counts'))
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


