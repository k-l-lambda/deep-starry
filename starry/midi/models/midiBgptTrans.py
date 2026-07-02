'''Conditioned-MIDI bGPT (TRANSLATOR variant): generate MIDI patches conditioned on a lilylet score.

Like starry.midi.models.condBgpt.CondMidiBGPT this is an AUTOREGRESSIVE bGPT model — the
next-patch target is the input patch ids shifted right by one, and a single token-level head
over the midi vocab decodes each predicted patch; ONLY the midi segment is supervised.

It DIFFERS from CondMidiBGPT in that the two modalities are TWO INDEPENDENT weight towers
(no shared backbone):

	- lyl_encoder: a `PatchLevelDecoder` (the SAME class/arch as LilyletNotaGen.patch_level_decoder)
	  encoding the lilylet score patches into per-patch memory. Because it is bit-for-bit the
	  NotaGen patch tower, its weights can be initialised from a pretrained LilyletNotaGen
	  checkpoint (`lyl_encoder_weights`) and optionally frozen (`freeze_lyl_encoder`).
	- midi decoder: a separate midi patch-embedding + a stack of transformer DecoderLayers doing
	  causal windowed self-attention over midi patches + cross-attention into the lyl memory, then
	  the midi token-level head. None of these weights are shared with the encoder.

The decoder-self (midi->midi window) and cross (midi->lyl window) masks are sliced from the
CondMidiPatchy joint `attn_mask`, so the measure coupling matches CondMidiBGPT. The lyl encoder
runs only over the lilylet segment (positions 0..max(lyl_counts)-1), keeping it cheap.

Consumes a CondMidiPatchy batch UNCHANGED:
	input_patches   LongTensor [B, T, patch_size]
	input_masks     LongTensor [B, T]               1 = real patch, 0 = padding
	input_targets   LongTensor [B, T]               1 = supervised midi target
	input_positions LongTensor [B, T]               per-modality 0..n-1
	attn_mask       BoolTensor [B, 1, T, T]          joint windowed/block mask (True = attend)
	modality        LongTensor [B, T]               0 = lilylet, 1 = midi
	lyl_counts      LongTensor [B]

Inference model (deducer): MidiBgptTrans
Loss model (training wrapper): MidiBgptTransLoss
'''

import math
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Config, LlamaConfig
from ...utils.registry import register_model
from ...transformer.layers import DecoderLayer
from ...bgpt.decoders import PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, PatchLevelDecoder, TokenLevelDecoder
from .bgpt import _space_token_id


def _build_patch_config (base_type, hidden_size, patch_num_layers, patch_length, n_head,
	intermediate_size=None, num_key_value_heads=None):
	'''Patch-level backbone config, identical to LilyletNotaGen's so weights load 1:1.'''
	n_head = n_head or max(1, hidden_size // 64)
	if base_type == 'llama':
		return LlamaConfig(
			num_hidden_layers=patch_num_layers, max_position_embeddings=patch_length,
			hidden_size=hidden_size, intermediate_size=intermediate_size or hidden_size * 4,
			num_attention_heads=n_head, num_key_value_heads=num_key_value_heads or n_head,
			vocab_size=1,
		)
	if base_type == 'gpt2':
		return GPT2Config(
			num_hidden_layers=patch_num_layers, max_length=patch_length,
			max_position_embeddings=patch_length, n_embd=hidden_size,
			num_attention_heads=n_head, vocab_size=1,
		)
	raise ValueError(f'Unknown base_type "{base_type}" (expected "gpt2" or "llama")')


def _load_lyl_encoder_weights (encoder, path):
	'''Load a pretrained LilyletNotaGen's patch tower into `encoder` (a PatchLevelDecoder).

	Accepts either a trainer checkpoint ({'model': deducer_state_dict, ...}) or a raw
	state_dict; pulls the `patch_level_decoder.*` sub-tree and strips that prefix so the keys
	align with `encoder.state_dict()`.
	'''
	blob = torch.load(path, map_location='cpu', weights_only=False)
	state = blob.get('model', blob) if isinstance(blob, dict) else blob
	prefix = 'patch_level_decoder.'
	sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
	if not sub:		# maybe already a bare patch-tower state_dict
		sub = state
	missing, unexpected = encoder.load_state_dict(sub, strict=False)
	return len(sub), list(missing), list(unexpected)


@register_model
class MidiBgptTrans (nn.Module):
	'''Inference model: independent lyl patch encoder + midi cross-attention decoder + midi head.

	Args (from config['model.args']):
		lyl_vocab_size / midi_vocab_size: tokenizer vocabs (256 / 41)
		patch_size: tokens per patch (16)
		--- lyl encoder (match the pretrained LilyletNotaGen patch tower for weight loading) ---
		lyl_base_type, lyl_hidden_size, lyl_patch_num_layers, lyl_patch_length, lyl_n_head,
		lyl_intermediate_size, lyl_num_key_value_heads
		lyl_encoder_weights: optional path to a LilyletNotaGen checkpoint to init the encoder
		freeze_lyl_encoder:  if True, the encoder is frozen (no grad, kept in eval)
		--- midi decoder ---
		d_model: midi decoder / token-head hidden size
		n_dec_layer: midi decoder layers (self + cross attention)
		token_num_layers: token-level (Llama) head layers
		n_head, d_k, d_v, d_inner, dropout: midi decoder attention dims
		num_key_value_heads: token-level Llama GQA kv heads (default n_head)
	'''

	def __init__ (self, lyl_vocab_size=256, midi_vocab_size=41, patch_size=16,
		lyl_base_type='llama', lyl_hidden_size=512, lyl_patch_num_layers=8, lyl_patch_length=1024,
		lyl_n_head=8, lyl_intermediate_size=2048, lyl_num_key_value_heads=8,
		lyl_encoder_weights=None, freeze_lyl_encoder=False,
		d_model=768, n_dec_layer=6, token_num_layers=3, n_head=12,
		d_k=None, d_v=None, d_inner=None, num_key_value_heads=None, dropout=0.1, **_):
		super().__init__()
		self.lyl_vocab_size = lyl_vocab_size
		self.midi_vocab_size = midi_vocab_size
		self.patch_size = patch_size
		self.d_model = d_model
		self.freeze_lyl_encoder = freeze_lyl_encoder
		self.special_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.eos_token_id = EOS_TOKEN_ID

		d_k = d_k or d_model // n_head
		d_v = d_v or d_model // n_head
		d_inner = d_inner or d_model * 4
		n_kv = num_key_value_heads or n_head

		# --- lyl encoder: a PatchLevelDecoder (same arch as LilyletNotaGen.patch_level_decoder) ---
		lyl_config = _build_patch_config(lyl_base_type, lyl_hidden_size, lyl_patch_num_layers,
			lyl_patch_length, lyl_n_head, lyl_intermediate_size, lyl_num_key_value_heads)
		self.lyl_encoder = PatchLevelDecoder(lyl_config, patch_size, lyl_vocab_size)
		self.lyl_hidden_size = lyl_hidden_size

		if lyl_encoder_weights and os.path.exists(lyl_encoder_weights):
			n, missing, unexpected = _load_lyl_encoder_weights(self.lyl_encoder, lyl_encoder_weights)
			print(f'[MidiBgptTrans] loaded {n} lyl-encoder tensors from {lyl_encoder_weights}'
				+ (f' (missing {len(missing)}, unexpected {len(unexpected)})' if missing or unexpected else ''))
		elif lyl_encoder_weights:
			print(f'[MidiBgptTrans] WARNING: lyl_encoder_weights not found: {lyl_encoder_weights}')

		if freeze_lyl_encoder:
			for p in self.lyl_encoder.parameters():
				p.requires_grad = False
			self.lyl_encoder.eval()

		# project lyl memory to the midi decoder width when the two hidden sizes differ.
		self.enc_proj = nn.Linear(lyl_hidden_size, d_model) if lyl_hidden_size != d_model else None
		if self.enc_proj is not None:
			nn.init.normal_(self.enc_proj.weight, std=0.02)

		# --- midi decoder: own patch-embedding + cross-attention decoder + token head ---
		self.midi_embedding = nn.Linear(patch_size * midi_vocab_size, d_model)
		nn.init.normal_(self.midi_embedding.weight, std=0.02)
		inv_freq = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		self.register_buffer('inv_freq', inv_freq, persistent=False)
		self.dropout = nn.Dropout(dropout)
		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
		self.decoder = nn.ModuleList([
			DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_dec_layer)])

		token_config = LlamaConfig(
			num_hidden_layers=token_num_layers, max_position_embeddings=patch_size + 1,
			hidden_size=d_model, intermediate_size=d_inner,
			num_attention_heads=n_head, num_key_value_heads=n_kv, vocab_size=midi_vocab_size,
		)
		self.token_level_decoder = TokenLevelDecoder(token_config)

	def train (self, mode=True):
		'''Keep a frozen lyl encoder in eval mode (dropout off) even under model.train().'''
		super().train(mode)
		if self.freeze_lyl_encoder:
			self.lyl_encoder.eval()
		return self

	def parameters_trainable (self):
		'''Parameters excluding a frozen lyl encoder, selected by MODULE IDENTITY (not the live
		`requires_grad` flag). The distributed validator calls `model.requires_grad_(False)` before
		the per-epoch param broadcast, which would zero a flag-based filter and desync the two ranks
		(the trainer keeps grads on) -> a gloo/NCCL collective size mismatch. Selecting by identity
		yields the SAME tensor list in the SAME order on both ranks, so broadcastParam stays aligned.
		'''
		if not self.freeze_lyl_encoder:
			return list(self.parameters())
		frozen_ids = {id(p) for p in self.lyl_encoder.parameters()}
		return [p for p in self.parameters() if id(p) not in frozen_ids]

	def _midi_embed (self, patches):
		dtype = self.midi_embedding.weight.dtype
		midi_ids = patches.clamp(max=self.midi_vocab_size - 1)
		oh = F.one_hot(midi_ids.long(), self.midi_vocab_size).to(dtype)
		return self.midi_embedding(oh.reshape(len(patches), -1, self.patch_size * self.midi_vocab_size))

	def _add_position (self, embeds, positions):
		ang = positions.unsqueeze(-1).float() * self.inv_freq			# [B,T,d/2]
		pe = torch.zeros_like(embeds)
		pe[..., 0::2] = torch.sin(ang)
		pe[..., 1::2] = torch.cos(ang)
		return embeds + pe

	def _dec_masks (self, attn_mask, modality, masks, Lmax):
		'''Derive the midi decoder's self + cross masks (True = attend).

		dec_self [B,T,T]: midi->midi window (sliced from the joint mask); non-midi query rows
		                  get a self-loop diagonal so SDPA softmax stays finite (rows discarded).
		cross    [B,T,Lmax]: midi->lyl window over the lyl-segment key positions 0..Lmax-1.
		'''
		joint = attn_mask.squeeze(1)									# [B,T,T]
		B, T = modality.shape
		q_midi = (modality == 1).unsqueeze(-1)							# [B,T,1]
		k_midi = (modality == 1).unsqueeze(1)							# [B,1,T]

		dec_self = joint & q_midi & k_midi
		eye = torch.eye(T, dtype=torch.bool, device=modality.device).unsqueeze(0)
		dec_self = dec_self | (~q_midi.expand(B, T, T) & eye)

		k_lyl_real = ((modality == 0) & (masks == 1)).unsqueeze(1)		# [B,1,T]
		cross = joint & q_midi & k_lyl_real								# [B,T,T]
		return dec_self, cross[:, :, :Lmax]

	def forward (self, patches, masks, modality, attn_mask, target_masks=None,
		positions=None, lyl_counts=None):
		'''
		Returns (token_output, target_patches): token-level next-patch prediction over the
		selected midi target patches (N = #targets across batch).
		'''
		patches = patches.reshape(len(patches), -1, self.patch_size)
		B, T, _ = patches.shape
		Lmax = int(lyl_counts.max().item()) if lyl_counts is not None else T

		# --- lyl encoder over the lilylet segment (positions 0..Lmax-1) ---
		lyl_patches = patches[:, :Lmax]
		lyl_real = ((modality[:, :Lmax] == 0) & (masks[:, :Lmax] == 1)).long()	# padding mask -> causal inside
		lyl_pos = positions[:, :Lmax] if positions is not None else None
		enc_ctx = torch.no_grad() if self.freeze_lyl_encoder else torch.enable_grad()
		with enc_ctx:
			memory = self.lyl_encoder(lyl_patches, lyl_real, position_ids=lyl_pos)['last_hidden_state']	# [B,Lmax,lyl_hidden]
		if self.enc_proj is not None:
			memory = self.enc_proj(memory)

		# --- midi decoder: embed midi patches, cross-attend into the score memory ---
		embeds = self._midi_embed(patches)								# [B,T,d_model]
		if positions is not None:
			embeds = self._add_position(embeds, positions)
		dec = self.layer_norm(self.dropout(embeds))

		dec_self, cross = self._dec_masks(attn_mask, modality, masks, Lmax)
		for layer in self.decoder:
			dec, _, _ = layer(dec, memory, slf_attn_mask=dec_self, dec_enc_attn_mask=cross)

		if target_masks is None:
			# Fallback (direct forward / inspect / inference): supervise the midi segment but drop
			# the FIRST midi patch per row. That patch sits at lyl_count (NOT index 0 — index 0 is
			# lilylet), so mirror CondMidiPatchy: never train the lilylet hidden state to emit the
			# first midi patch across the modality boundary.
			target_masks = (modality == 1).long() * masks
			if lyl_counts is not None:
				rows = torch.arange(B, device=patches.device)
				has_midi = lyl_counts < T
				target_masks[rows[has_midi], lyl_counts[has_midi]] = 0
			else:
				# no lyl_counts: fall back to dropping the first real midi patch of each row.
				first_midi = (target_masks == 1).float().argmax(dim=1)		# 0 if a row has none
				has_midi = target_masks.sum(dim=1) > 0
				rows = torch.arange(B, device=patches.device)
				target_masks[rows[has_midi], first_midi[has_midi]] = 0
		else:
			target_masks = target_masks.clone()

		# Next-patch prediction: decoded patch i predicts patch i+1. An encoded position is a
		# supervised INPUT iff its NEXT patch is a target; gate by the real-patch mask.
		left_shift = torch.zeros_like(masks)
		left_shift[:, :-1] = target_masks[:, 1:]
		left_shift = left_shift * masks

		# Degenerate guard: if a batch ends up with NO supervised target (e.g. a tiny sample whose
		# only midi content is the unsupervised header, or every body target dropped by the crop),
		# `dec_sel`/`target_patches` would be zero-length and crash TokenLevelDecoder's HF LM. Fall
		# back to the last adjacent real-midi (context, target) pair so the loss stays finite; this
		# fires only on pathological samples (normal crops keep >=1 whole body measure).
		if int(left_shift.sum()) == 0 or int(target_masks.sum()) == 0:
			real_midi = (modality == 1) & (masks == 1)					# [B,T]
			flat = real_midi.reshape(-1)
			pos = torch.nonzero(flat, as_tuple=False).flatten()
			if pos.numel() >= 2 and int(pos[-1]) - int(pos[-2]) == 1:	# adjacent in the flat layout
				tgt_flat = int(pos[-1]); ctx_flat = int(pos[-2])
				left_shift = torch.zeros_like(masks); target_masks = torch.zeros_like(masks)
				left_shift.reshape(-1)[ctx_flat] = 1
				target_masks.reshape(-1)[tgt_flat] = 1

		dec_sel = dec[left_shift == 1]					# [N, d_model]
		target_patches = patches[target_masks == 1]		# [N, patch_size]
		return self.token_level_decoder(dec_sel, target_patches), target_patches


@register_model
class MidiBgptTransLoss (nn.Module):
	'''Training wrapper: loss + metrics from a CondMidiPatchy batch (midi-segment only).'''

	def __init__ (self, **kw_args):
		super().__init__()
		self.deducer = MidiBgptTrans(**kw_args)

	def training_parameters (self):
		# exclude a frozen lyl encoder so the optimizer never sees zero-grad params.
		return self.deducer.parameters_trainable() + list(self.deducer.buffers())

	def validation_parameters (self):
		return []

	def _token_accuracy (self, output, target_patches):
		'''Next-token accuracy over valid (non-pad) midi token positions.'''
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
