'''Conditioned-MIDI bGPT (SELF-ATTENTION TRANSLATOR variant): generate MIDI patches conditioned
on a lilylet score, using a SINGLE self-attention decoder over the JOINT sequence instead of the
cross-attention decoder of MidiBgptTrans.

Motivation: MidiBgptTrans keeps two independent towers coupled by hand-written cross-attention
(`starry/transformer/layers.DecoderLayer` + the custom `ScaledDotProductAttention`), which
materializes a dense `[B, n_head, T, T]` score matrix and is O(T^2) in memory/compute regardless
of the windowed mask. This variant keeps the SAME independent-tower reuse (a pretrained, optionally
frozen `PatchLevelDecoder` lyl encoder) but folds the conditioning into a prefix-LM: the encoder's
per-patch hidden states are PROJECTED to the decoder width and SCATTERED into the lilylet-prefix
positions of a joint input sequence, whose midi positions carry the midi patch embedding. One HF
`LlamaModel` then runs self-attention over the whole sequence under the joint `[B,1,T,T]` windowed
mask (the SAME mask CondMidiPatchy already builds), so it inherits HF's fused SDPA / FlexAttention
path (no dense score matrix) and RoPE positions for free.

Compared to the two siblings:
  - CondMidiBGPT (DualPatchLevelDecoder): shared backbone, re-embeds lyl from raw patches -> cannot
    load+freeze a pretrained LilyletNotaGen encoder.
  - MidiBgptTrans: independent frozen encoder + cross-attention decoder -> reuses the encoder but
    pays the O(T^2) hand-rolled attention.
  - MidiBgptTransSelfAttn (this): independent frozen encoder AS A PROJECTED PREFIX + single HF
    self-attention decoder -> reuses the encoder AND gets the fused-kernel efficiency.

This is NOT numerically equivalent to MidiBgptTrans (cross-attention is gone); the decoder trains
from scratch. Only the lyl encoder weights transfer.

Consumes a CondMidiPatchy batch UNCHANGED (identical contract to MidiBgptTrans):
	input_patches   LongTensor [B, T, patch_size]
	input_masks     LongTensor [B, T]               1 = real patch, 0 = padding
	input_targets   LongTensor [B, T]               1 = supervised midi target
	input_positions LongTensor [B, T]               per-modality 0..n-1
	attn_mask       BoolTensor [B, 1, T, T]          joint windowed/block mask (True = attend)
	modality        LongTensor [B, T]               0 = lilylet, 1 = midi
	lyl_counts      LongTensor [B]

Inference model (deducer): MidiBgptTransSelfAttn
Loss model (training wrapper): MidiBgptTransSelfAttnLoss
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel
from ...utils.registry import register_model
from ...bgpt.decoders import PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, PatchLevelDecoder, TokenLevelDecoder
from .bgpt import _space_token_id
from .midiBgptTrans import _build_patch_config, _load_lyl_encoder_weights


@register_model
class MidiBgptTransSelfAttn (nn.Module):
	'''Inference model: independent lyl patch encoder (projected prefix) + single self-attention
	joint decoder + midi token head.

	Args (from config['model.args']):
		lyl_vocab_size / midi_vocab_size: tokenizer vocabs (256 / 41)
		patch_size: tokens per patch (16)
		--- lyl encoder (match the pretrained LilyletNotaGen patch tower for weight loading) ---
		lyl_base_type, lyl_hidden_size, lyl_patch_num_layers, lyl_patch_length, lyl_n_head,
		lyl_intermediate_size, lyl_num_key_value_heads
		lyl_encoder_weights: optional path to a LilyletNotaGen checkpoint to init the encoder
		freeze_lyl_encoder:  if True, the encoder is frozen (no grad, kept in eval)
		--- joint self-attention decoder (HF Llama backbone; inherits fused SDPA) ---
		d_model: decoder / token-head hidden size
		n_dec_layer: decoder self-attention layers
		token_num_layers: token-level (Llama) head layers
		n_head, d_inner, dropout: decoder attention dims
			dec_num_key_value_heads: decoder self-attn GQA kv heads (default n_head = full MHA)
			num_key_value_heads: token-level (Llama) head GQA kv heads (default n_head)
	'''

	def __init__ (self, lyl_vocab_size=256, midi_vocab_size=41, patch_size=16,
		lyl_base_type='llama', lyl_hidden_size=512, lyl_patch_num_layers=8, lyl_patch_length=1024,
		lyl_n_head=8, lyl_intermediate_size=2048, lyl_num_key_value_heads=8,
		lyl_encoder_weights=None, freeze_lyl_encoder=False,
		d_model=768, n_dec_layer=6, token_num_layers=3, n_head=12,
		d_inner=None, dec_num_key_value_heads=None, num_key_value_heads=None, dropout=0.1, **_):
		super().__init__()
		self.lyl_vocab_size = lyl_vocab_size
		self.midi_vocab_size = midi_vocab_size
		self.patch_size = patch_size
		self.d_model = d_model
		self.freeze_lyl_encoder = freeze_lyl_encoder
		self.special_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.eos_token_id = EOS_TOKEN_ID

		d_inner = d_inner or d_model * 4
		# decoder self-attention: default to full MHA (kv == heads). token head: its own GQA param.
		dec_n_kv = dec_num_key_value_heads or n_head
		n_kv = num_key_value_heads or n_head
		assert n_head % dec_n_kv == 0, \
			f'n_head ({n_head}) must be divisible by dec_num_key_value_heads ({dec_n_kv})'
		assert n_head % n_kv == 0, \
			f'n_head ({n_head}) must be divisible by num_key_value_heads ({n_kv})'

		# --- lyl encoder: a PatchLevelDecoder (same arch as LilyletNotaGen.patch_level_decoder) ---
		lyl_config = _build_patch_config(lyl_base_type, lyl_hidden_size, lyl_patch_num_layers,
			lyl_patch_length, lyl_n_head, lyl_intermediate_size, lyl_num_key_value_heads)
		self.lyl_encoder = PatchLevelDecoder(lyl_config, patch_size, lyl_vocab_size)
		self.lyl_hidden_size = lyl_hidden_size

		if lyl_encoder_weights and os.path.exists(lyl_encoder_weights):
			n, missing, unexpected = _load_lyl_encoder_weights(self.lyl_encoder, lyl_encoder_weights)
			print(f'[MidiBgptTransSelfAttn] loaded {n} lyl-encoder tensors from {lyl_encoder_weights}'
				+ (f' (missing {len(missing)}, unexpected {len(unexpected)})' if missing or unexpected else ''))
		elif lyl_encoder_weights:
			print(f'[MidiBgptTransSelfAttn] WARNING: lyl_encoder_weights not found: {lyl_encoder_weights}')

		if freeze_lyl_encoder:
			for p in self.lyl_encoder.parameters():
				p.requires_grad = False
			self.lyl_encoder.eval()

		# project lyl memory to the decoder width (always present; identity-free Linear even when
		# widths match keeps the prefix trainable given the encoder may be frozen).
		self.enc_proj = nn.Linear(lyl_hidden_size, d_model)
		nn.init.normal_(self.enc_proj.weight, std=0.02)

		# --- midi patch embedding (own tower; lyl positions get the projected encoder prefix) ---
		self.midi_embedding = nn.Linear(patch_size * midi_vocab_size, d_model)
		nn.init.normal_(self.midi_embedding.weight, std=0.02)

		# --- joint self-attention decoder: one HF Llama backbone over [prefix ; midi] ---
		dec_config = LlamaConfig(
			num_hidden_layers=n_dec_layer, max_position_embeddings=lyl_patch_length,
			hidden_size=d_model, intermediate_size=d_inner,
			num_attention_heads=n_head, num_key_value_heads=dec_n_kv, vocab_size=1,
			attention_dropout=dropout,
		)
		self.decoder = LlamaModel(dec_config)

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
		`requires_grad` flag) so the distributed validator (which calls requires_grad_(False)
		before the per-epoch param broadcast) yields the SAME tensor list/order as the trainer.
		See MidiBgptTrans.parameters_trainable for the gloo/NCCL size-mismatch rationale.
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

	def forward (self, patches, masks, modality, attn_mask, target_masks=None,
		positions=None, lyl_counts=None):
		'''
		Returns (token_output, target_patches): token-level next-patch prediction over the
		selected midi target patches (N = #targets across batch).
		'''
		patches = patches.reshape(len(patches), -1, self.patch_size)
		B, T, _ = patches.shape
		Lmax = int(lyl_counts.max().item()) if lyl_counts is not None else 0

		# --- lyl encoder over the lilylet prefix (positions 0..Lmax-1) ---
		# lyl is always the contiguous prefix [0, lyl_count), so the encoder memory [B,Lmax,.]
		# aligns 1:1 with the first Lmax sequence positions.
		if Lmax > 0:
			lyl_patches = patches[:, :Lmax]
			lyl_real = ((modality[:, :Lmax] == 0) & (masks[:, :Lmax] == 1)).long()
			lyl_pos = positions[:, :Lmax] if positions is not None else None
			enc_ctx = torch.no_grad() if self.freeze_lyl_encoder else torch.enable_grad()
			with enc_ctx:
				memory = self.lyl_encoder(lyl_patches, lyl_real, position_ids=lyl_pos)['last_hidden_state']
			memory = self.enc_proj(memory)								# [B,Lmax,d_model]

		# --- joint embedding: projected encoder prefix on lyl positions, midi embedding on midi ---
		embeds = self._midi_embed(patches)								# [B,T,d_model]
		if Lmax > 0:
			is_lyl = (modality[:, :Lmax] == 0).unsqueeze(-1)			# [B,Lmax,1]
			embeds[:, :Lmax] = torch.where(is_lyl, memory.to(embeds.dtype), embeds[:, :Lmax])

		# --- single self-attention decoder over the joint sequence under the windowed mask ---
		# The 4D bool attn_mask [B,1,T,T] is passed straight to the HF Llama backbone (True =
		# attend), exactly as DualPatchLevelDecoder does; RoPE uses the per-modality positions.
		dec = self.decoder(inputs_embeds=embeds, attention_mask=attn_mask,
			position_ids=positions)['last_hidden_state']				# [B,T,d_model]

		if target_masks is None:
			# Fallback (direct forward / inspect / inference): supervise the midi segment but drop
			# the FIRST midi patch per row (at lyl_count) — mirror CondMidiPatchy: never train the
			# lilylet hidden state to emit the first midi patch across the modality boundary.
			target_masks = (modality == 1).long() * masks
			if lyl_counts is not None:
				rows = torch.arange(B, device=patches.device)
				has_midi = lyl_counts < T
				target_masks[rows[has_midi], lyl_counts[has_midi]] = 0
			else:
				first_midi = (target_masks == 1).float().argmax(dim=1)
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

		# Degenerate guard (see MidiBgptTrans.forward): rebuild a single (context, target) pair
		# from the LAST real-midi patch(es) so TokenLevelDecoder never sees a 0-size batch. The
		# target MUST be a real midi patch (ids in the midi vocab, not the 0..255 lyl range).
		if int(left_shift.sum()) == 0 or int(target_masks.sum()) == 0:
			real_midi = (modality == 1) & (masks == 1)
			pos = torch.nonzero(real_midi.reshape(-1), as_tuple=False).flatten()
			if pos.numel() >= 1:
				tgt_flat = int(pos[-1])
				ctx_flat = int(pos[-2]) if pos.numel() >= 2 else tgt_flat
				left_shift = torch.zeros_like(masks); target_masks = torch.zeros_like(masks)
				left_shift.reshape(-1)[ctx_flat] = 1
				target_masks.reshape(-1)[tgt_flat] = 1

		dec_sel = dec[left_shift == 1]					# [N, d_model]
		target_patches = patches[target_masks == 1]		# [N, patch_size]
		return self.token_level_decoder(dec_sel, target_patches), target_patches


@register_model
class MidiBgptTransSelfAttnLoss (nn.Module):
	'''Training wrapper: loss + metrics from a CondMidiPatchy batch (midi-segment only).'''

	def __init__ (self, **kw_args):
		super().__init__()
		self.deducer = MidiBgptTransSelfAttn(**kw_args)

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
