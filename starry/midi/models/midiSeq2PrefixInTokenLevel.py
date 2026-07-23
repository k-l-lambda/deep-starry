'''Conditioned-MIDI bGPT — lyl-conditioning IN THE TOKEN LEVEL (midiseq2 SPLIT-MODALITY variant).

Consumes the SPLIT-MODALITY batch (starry.midi.data.seq2CondSplitPatchy.Seq2CondSplitMidiPatchy):
lilylet and midi arrive as SEPARATE tensors at their native widths / vocabularies, so there is no
dual-width [T,64] frame, no shared integer axis and no id clamp. The lilylet conditioning lives in
the TOKEN-LEVEL decoder rather than a joint decoder:

  - A MIDI-ONLY self-attention decoder runs over the midi patches under the midi->midi window mask
    (midi_attn_mask, the midi block of build_vis); each midi patch's hidden state is built purely
    from the midi->midi window (w_midi). There is NO lyl encoder and NO lyl memory scatter.
  - The lilylet signal enters at the TOKEN decoder: for each supervised midi target patch, the lyl
    patches in ITS cross window — derived directly from measure ids (midi_src / lyl_meas / w_cross),
    NOT a mask tensor — are re-embedded from their raw lyl ids (a fresh lyl_prefix_embedding Linear,
    one-hot over lyl_vocab_size) and PREPENDED as prefix patches ahead of [encoded_patch, tokens...]
    inside a PrefixTokenLevelDecoder, carrying NEGATIVE, INCREASING RoPE positions (-Kmax..-1). This
    gives the within-patch token generator direct access to exactly the relevant lilylet measures.

This deducer is a plain nn.Module (it does NOT subclass MidiSeq2BgptSelfAttn): the pretrained
LilyletNotaGen encoder / enc_proj / dual-width machinery of that model are simply not needed here, so
this class builds only what it uses — a midi embedding, a midi self-attention LlamaModel decoder, a
PrefixTokenLevelDecoder token head, and the lyl-prefix embedding.

The Loss (MidiSeq2PrefixInTokenLevelLoss) DOES reuse MidiSeq2BgptSelfAttnLoss's loss/metric machinery
(weighted CE + per-token-type err_*, stat()); the token output exposes prefix-sliced logits
[N, patch_size+1, vocab] so every helper works unchanged.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel

from ...utils.registry import register_model
from ...bgpt.decoders import PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, TokenLevelDecoder, token_embedding_weight
from .midiSeq2BgptSelfAttn import (
	MidiSeq2BgptSelfAttnLoss,
	_TYPE_CODES, _TYPE_NAMES, _build_type_map,
)
from ..data.seq2CondPachifier import Midiseq2Tokenizer


class PrefixTokenLevelDecoder (TokenLevelDecoder):
	'''TokenLevelDecoder that prepends a variable number of PREFIX embeddings ahead of the standard
	[encoded_patch, tok_1..tok_P] token sequence. Prefixes carry negative, increasing positions
	(-Kmax..-1); the encoded patch is position 0; the P teacher-forced tokens are positions 1..P.

	Prefix slots are never supervised (labels -100) and are excluded from the returned logits so
	downstream loss/metric code sees the SAME [N, patch_size+1, vocab] shape as the stock decoder.
	With Kmax == 0 this reduces exactly to TokenLevelDecoder.forward (single encoded slot at 0).
	'''

	def forward (self, encoded_patches, target_patches, prefix_embeds=None, prefix_mask=None):
		'''
		encoded_patches: FloatTensor [N, hidden]              per-target patch hidden state
		target_patches:  LongTensor  [N, P]                   token ids to teacher-force (P = patch_size)
		prefix_embeds:   FloatTensor [N, Kmax, hidden] | None  re-embedded lyl cross-window patches
		prefix_mask:     LongTensor  [N, Kmax] | None          1 = real prefix, 0 = left-pad slot
		Returns: HF CausalLM output; .logits SLICED to [N, P+1, vocab] (prefix stripped), .loss over
			the P supervised token positions (prefix/encoded excluded via -100 labels).
		'''
		N = target_patches.shape[0]
		# BOS-prepend then build token embeddings; slot 0's embedding is overwritten by encoded below.
		bos_tokens = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.bos_token_id, target_patches), dim=1)	# [N,P+1]
		labels = bos_tokens.masked_fill(bos_tokens == self.special_token_id, -100)
		token_masks = (labels != -100).long()														# [N,P+1]														# [N,P+1]

		tok_embeds = F.embedding(bos_tokens, token_embedding_weight(self.base))						# [N,P+1,h]
		midi_embeds = torch.cat((encoded_patches.unsqueeze(1), tok_embeds[:, 1:, :]), dim=1)			# [N,P+1,h]

		P1 = midi_embeds.shape[1]			# P + 1
		device = midi_embeds.device
		Kmax = prefix_embeds.shape[1] if prefix_embeds is not None else 0

		if Kmax == 0:
			# Exact stock-decoder path: encoded at position 0, tokens 1..P, HF default positions
			# and HF-added causal mask (byte-identical to TokenLevelDecoder.forward).
			return self.base(inputs_embeds=midi_embeds, attention_mask=token_masks, labels=labels)

		# The encoded patch at position 0 is an input context, not a supervised output.
		# With a prefix, leaving labels[:, 0] as BOS would train the final prefix logit
		# to predict BOS after HF's causal shift.
		labels[:, 0] = -100
		token_masks = (labels != -100).long()

		inputs_embeds = torch.cat((prefix_embeds.to(midi_embeds.dtype), midi_embeds), dim=1)			# [N,K+P+1,h]
		pmask = prefix_mask.long() if prefix_mask is not None else torch.ones(N, Kmax, dtype=torch.long, device=device)
		key_valid = torch.cat((pmask, token_masks), dim=1).bool()									# [N,K+P+1]
		prefix_labels = torch.full((N, Kmax), -100, dtype=labels.dtype, device=device)
		full_labels = torch.cat((prefix_labels, labels), dim=1)										# [N,K+P+1]

		# Explicit 4D bool mask (True = attend), like the feeder: causal AND key-not-pad, plus a forced
		# self-diagonal so a left-pad prefix query row is never fully masked (which would NaN the softmax
		# and — since a position is a key at the next layer — propagate). Padded keys stay masked for
		# every REAL query, so the self-diagonal only well-defines the (discarded) padded-row outputs.
		S = key_valid.shape[1]
		idx = torch.arange(S, device=device)
		causal = idx.unsqueeze(0) <= idx.unsqueeze(1)												# [S,S] key<=query
		mask4d = (causal.unsqueeze(0) & key_valid.unsqueeze(1))										# [N,S,S]
		mask4d = mask4d | torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0)
		mask4d = mask4d.unsqueeze(1)
		# Use additive form: HF eager attention adds 4D masks to scores, while SDPA
		# interprets bool masks differently. Additive masking is backend-independent.
		mask4d = torch.zeros_like(mask4d, dtype=midi_embeds.dtype).masked_fill(
			~mask4d, torch.finfo(midi_embeds.dtype).min)

		# positions: prefixes -Kmax..-1 (minus, increasing, most-recent lyl at -1), encoded 0, tokens 1..P.
		prefix_pos = torch.arange(-Kmax, 0, device=device)											# [-Kmax..-1]
		midi_pos = torch.arange(0, P1, device=device)												# [0..P]
		position_ids = torch.cat((prefix_pos, midi_pos)).unsqueeze(0).expand(N, -1)					# [N,K+P+1]

		out = self.base(inputs_embeds=inputs_embeds, attention_mask=mask4d,
			position_ids=position_ids, labels=full_labels)
		# strip the prefix positions from logits so downstream code sees [N, P+1, vocab].
		out.logits = out.logits[:, Kmax:, :]
		return out


@register_model
class MidiSeq2PrefixInTokenLevel (nn.Module):
	'''Deducer: a MIDI-only self-attention decoder + per-target lyl-cross-window prefix in the token
	decoder, over the SPLIT-MODALITY batch. Plain nn.Module — builds only what it uses (midi
	embedding, midi self-attention LlamaModel, PrefixTokenLevelDecoder token head, lyl-prefix Linear);
	no lyl encoder / enc_proj (the lilylet condition is re-embedded raw at the token level).
	'''

	def __init__ (self, lyl_vocab_size=256, midi_vocab_size=838, patch_size=64, lyl_patch_size=16,
		lyl_patch_length=1024, w_cross=1,
		d_model=768, n_dec_layer=6, token_num_layers=3, n_head=12,
		d_inner=None, dec_num_key_value_heads=None, num_key_value_heads=None, dropout=0.1, **_):
		super().__init__()
		self.lyl_vocab_size = lyl_vocab_size
		self.midi_vocab_size = midi_vocab_size
		self.patch_size = patch_size
		self.lyl_patch_size = lyl_patch_size
		self.d_model = d_model
		# midi->lyl cross window (in MEASURES) used to gather each target patch's lyl prefix. Default 1
		# (a midi patch sees only its aligned lyl measure + the always-on lyl header).
		self.w_cross = w_cross
		self.special_token_id = PAD_TOKEN_ID
		self.bos_token_id = BOS_TOKEN_ID
		self.eos_token_id = EOS_TOKEN_ID

		d_inner = d_inner or d_model * 4
		dec_n_kv = dec_num_key_value_heads or n_head
		n_kv = num_key_value_heads or n_head
		assert n_head % dec_n_kv == 0, \
			f'n_head ({n_head}) must be divisible by dec_num_key_value_heads ({dec_n_kv})'
		assert n_head % n_kv == 0, \
			f'n_head ({n_head}) must be divisible by num_key_value_heads ({n_kv})'

		# --- midi patch embedding (one-hot over midi vocab -> d_model) ---
		self.midi_embedding = nn.Linear(patch_size * midi_vocab_size, d_model)
		nn.init.normal_(self.midi_embedding.weight, std=0.02)

		# --- raw-lyl re-embedding for the token-level prefix (one-hot over lyl vocab -> d_model) ---
		self.lyl_prefix_embedding = nn.Linear(lyl_patch_size * lyl_vocab_size, d_model)
		nn.init.normal_(self.lyl_prefix_embedding.weight, std=0.02)

		# --- midi self-attention decoder: one HF Llama backbone over the midi patches ---
		dec_config = LlamaConfig(
			num_hidden_layers=n_dec_layer, max_position_embeddings=lyl_patch_length,
			hidden_size=d_model, intermediate_size=d_inner,
			num_attention_heads=n_head, num_key_value_heads=dec_n_kv, vocab_size=1,
			attention_dropout=dropout,
		)
		self.decoder = LlamaModel(dec_config)

		# --- token head: multi-prefix decoder. max_position_embeddings must cover Kmax + P + 1 and
		# the negative prefix positions; lyl_patch_length (1024) is generous. ---
		token_config = LlamaConfig(
			num_hidden_layers=token_num_layers, max_position_embeddings=lyl_patch_length,
			hidden_size=d_model, intermediate_size=d_inner,
			num_attention_heads=n_head, num_key_value_heads=n_kv, vocab_size=midi_vocab_size,
		)
		self.token_level_decoder = PrefixTokenLevelDecoder(token_config)

	def parameters_trainable (self):
		return list(self.parameters())

	def _midi_embed (self, patches):
		dtype = self.midi_embedding.weight.dtype
		oh = F.one_hot(patches.long(), self.midi_vocab_size).to(dtype)
		return self.midi_embedding(oh.reshape(len(patches), -1, self.patch_size * self.midi_vocab_size))

	def _embed_lyl_prefix (self, lyl_ids):
		'''One-hot the lyl prefix patch ids over lyl_vocab and project to d_model. lyl_ids: [N,K,lp].'''
		dtype = self.lyl_prefix_embedding.weight.dtype
		N, K, lp = lyl_ids.shape
		oh = F.one_hot(lyl_ids.long(), self.lyl_vocab_size).to(dtype)			# [N,K,lp,lyl_vocab]
		return self.lyl_prefix_embedding(oh.reshape(N, K, lp * self.lyl_vocab_size))	# [N,K,d_model]

	def forward (self, lyl_patches, lyl_masks, lyl_meas, midi_patches, midi_masks, midi_meas,
		midi_src, midi_attn_mask, midi_targets=None, midi_positions=None):
		'''Midi-only self-attention decoder + per-target lyl-cross-window prefix in the token decoder,
		over the SPLIT batch. Returns (token_output, target_patches).
		'''
		midi_patches = midi_patches.reshape(len(midi_patches), -1, self.patch_size)
		B, Mp, _ = midi_patches.shape

		# --- midi embedding + midi->midi windowed self-attention decoder ---
		embeds = self._midi_embed(midi_patches)							# [B,Mp,d_model]
		dec = self.decoder(inputs_embeds=embeds, attention_mask=midi_attn_mask,
			position_ids=midi_positions)['last_hidden_state']			# [B,Mp,d_model]

		# --- target / left-shift selection (midi-frame only; drop the first midi patch) ---
		if midi_targets is None:
			target_masks = midi_masks.clone()
			has_midi = midi_masks.sum(dim=1) > 0
			first_midi = midi_masks.float().argmax(dim=1)
			rows = torch.arange(B, device=midi_patches.device)
			target_masks[rows[has_midi], first_midi[has_midi]] = 0
		else:
			target_masks = midi_targets.clone()

		left_shift = torch.zeros_like(midi_masks)
		left_shift[:, :-1] = target_masks[:, 1:]
		left_shift = left_shift * midi_masks

		# Degenerate guard: rebuild one (context, target) pair from the last real midi patch so the
		# token decoder never sees a 0-size batch.
		if int(left_shift.sum()) == 0 or int(target_masks.sum()) == 0:
			real_midi = midi_masks == 1
			pos = torch.nonzero(real_midi.reshape(-1), as_tuple=False).flatten()
			if pos.numel() >= 1:
				tgt_flat = int(pos[-1])
				ctx_flat = int(pos[-2]) if pos.numel() >= 2 else tgt_flat
				left_shift = torch.zeros_like(midi_masks); target_masks = torch.zeros_like(midi_masks)
				left_shift.reshape(-1)[ctx_flat] = 1
				target_masks.reshape(-1)[tgt_flat] = 1

		dec_sel = dec[left_shift == 1]					# [N, d_model]
		target_patches = midi_patches[target_masks == 1]	# [N, patch_size]

		# --- per-target lyl cross-window prefix (derived from measure ids, no cross-mask tensor) ---
		tgt_idx = torch.nonzero(target_masks == 1, as_tuple=False)		# [N,2] -> (b, p)
		prefix_embeds, prefix_mask = self._gather_lyl_prefix(
			lyl_patches, lyl_masks, lyl_meas, midi_src, tgt_idx)

		return self.token_level_decoder(dec_sel, target_patches, prefix_embeds, prefix_mask), target_patches

	def _gather_lyl_prefix (self, lyl_patches, lyl_masks, lyl_meas, midi_src, tgt_idx):
		'''For each target midi patch (b, p), select the lyl patches in its cross window via the
		build_vis cross rule computed FROM MEASURE IDS (no mask tensor), and re-embed them
		RIGHT-ALIGNED into [N, Kmax] (most-recent lyl at the last slot = position -1).

		Cross rule (w_cross measures, matching condPatchy.build_vis): a target with source measure
		s = midi_src[b,p] attends lyl key k iff lyl_masks and either
		  - lyl_meas[b,k] == 0            (lyl prompt/header — always visible), OR
		  - s != 0 and s - w_cross < lyl_meas[b,k] <= s   (the aligned measure + w_cross-1 before it).

		Returns (prefix_embeds [N,Kmax,d_model], prefix_mask [N,Kmax]) or (None, None) when no target
		attends any lyl patch (Kmax == 0 -> the token decoder falls back to the stock single-slot path).
		'''
		N = tgt_idx.shape[0]
		device = lyl_patches.device
		Lp = lyl_patches.shape[1]
		if N == 0 or Lp == 0:
			return None, None
		b_idx = tgt_idx[:, 0]
		p_idx = tgt_idx[:, 1]

		s = midi_src[b_idx, p_idx].unsqueeze(1)							# [N,1] target source measure
		lm = lyl_meas[b_idx]											# [N,Lp] lyl measure of each key
		real = lyl_masks[b_idx] == 1									# [N,Lp]
		header = lm == 0
		in_window = (s != 0) & (lm > s - self.w_cross) & (lm <= s)
		valid = real & (header | in_window)								# [N,Lp]
		counts = valid.sum(dim=1)										# [N]
		Kmax = int(counts.max().item())
		if Kmax == 0:
			return None, None

		# right-align: within each row, the r-th valid key (0-based) goes to slot Kmax-count+r.
		rank = valid.long().cumsum(dim=1) - 1							# [N,Lp], -1 where not counted yet
		slot = (Kmax - counts).unsqueeze(1) + rank						# [N,Lp] target slot for valid keys
		key_pos = torch.arange(Lp, device=device).unsqueeze(0).expand(N, -1)

		pos_grid = torch.zeros(N, Kmax, dtype=torch.long, device=device)	# lyl-index of each prefix slot
		prefix_mask = torch.zeros(N, Kmax, dtype=torch.long, device=device)
		flat_n = torch.arange(N, device=device).unsqueeze(1).expand(-1, Lp)[valid]
		flat_slot = slot[valid]
		pos_grid[flat_n, flat_slot] = key_pos[valid]
		prefix_mask[flat_n, flat_slot] = 1

		# gather the raw lyl patch ids at those lyl-indices (native lyl width, pure lyl vocab).
		lyl_ids = lyl_patches[b_idx.unsqueeze(1), pos_grid]				# [N,Kmax,lyl_patch_size]
		prefix_embeds = self._embed_lyl_prefix(lyl_ids)					# [N,Kmax,d_model]
		# zero out padded slots so they contribute nothing (also masked by prefix_mask in attention).
		prefix_embeds = prefix_embeds * prefix_mask.unsqueeze(-1).to(prefix_embeds.dtype)
		return prefix_embeds, prefix_mask


@register_model
class MidiSeq2PrefixInTokenLevelLoss (MidiSeq2BgptSelfAttnLoss):
	'''Training wrapper for MidiSeq2PrefixInTokenLevel. Identical loss/metric machinery to
	MidiSeq2BgptSelfAttnLoss (weighted CE + per-token-type err_*, stat()); only the deducer differs.
	'''

	def __init__ (self, loss_type_weights=None, **kw_args):
		nn.Module.__init__(self)
		self.deducer = MidiSeq2PrefixInTokenLevel(**kw_args)
		type_map = _build_type_map(Midiseq2Tokenizer())
		self.register_buffer('type_of_id', type_map, persistent=False)
		self.type_names = dict(_TYPE_NAMES)

		self.loss_type_weights = dict(loss_type_weights) if loss_type_weights else {}
		type_weight = torch.ones(len(_TYPE_CODES))
		for name, w in self.loss_type_weights.items():
			if name not in _TYPE_CODES:
				raise ValueError(f'unknown loss_type_weights key {name!r}; valid: {sorted(_TYPE_CODES)}')
			type_weight[_TYPE_CODES[name]] = float(w)
		weight_of_id = type_weight[type_map]
		self.register_buffer('ce_weight_of_id', weight_of_id, persistent=False)
		self.weighted_loss = any(float(w) != 1.0 for w in self.loss_type_weights.values())

	def _run_deducer (self, batch):
		'''Unpack the SPLIT batch keys and call the deducer -> (token_output, target_patches).'''
		return self.deducer(
			batch['lyl_patches'], batch['lyl_masks'], batch['lyl_meas'],
			batch['midi_patches'], batch['midi_masks'], batch['midi_meas'], batch['midi_src'],
			batch['midi_attn_mask'], batch.get('midi_targets'), batch.get('midi_positions'))

	def forward (self, batch):
		output, target = self._run_deducer(batch)
		loss = self._weighted_loss(output, target) if self.weighted_loss else output.loss
		with torch.no_grad():
			acc = self._token_accuracy(output, target)
			metrics = {'acc': acc, 'err': 1 - acc}
			if not self.training:
				metrics.update(self._grouped_error(output, target))
		return loss, metrics

	def inspectRun (self, batch):
		output, target = self._run_deducer(batch)
		acc = self._token_accuracy(output, target)
		loss = self._weighted_loss(output, target) if self.weighted_loss else output.loss
		metrics = {
			'loss': loss.item(),
			'acc': acc,
			'err': 1 - acc,
			'logits': output.logits,
			'target_patches': target,
			'n_patches': int(target.shape[0]),
		}
		metrics.update({k: v.value for k, v in self._grouped_error(output, target).items()})
		return metrics
