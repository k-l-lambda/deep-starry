'''Conditioned-MIDI bGPT SELF-ATTENTION TRANSLATOR — midiseq2 variant.

Derived from starry.midi.models.midiBgptTransSelfAttn.MidiBgptTransSelfAttn WITHOUT modifying it,
so the basic-MIDI path stays byte-compatible. The base shares a SINGLE patch_size (16) between the
lilylet encoder and the midi side; the midiseq2 feeder (starry.midi.data.seq2CondPatchy) is
DUAL-WIDTH instead: joint patches are patch_size=64 wide, with lilylet token ids in columns
[0, lyl_patch_size=16) (<pad> tail) and midiseq2 ids (vocab 582) using the full 64 columns.

This subclass fixes exactly that mismatch:
  - `lyl_encoder` is rebuilt at lyl_patch_size (16) so a pretrained LilyletNotaGen patch tower
    loads 1:1 (the base built it at 64, silently dropping the shape-mismatched patch_embedding).
  - `forward` feeds the encoder only the first lyl_patch_size columns of each lilylet patch.
Everything else (enc_proj, joint-embed overwrite, decoder, target selection, degenerate guard,
midi_embedding at the full 64 width) is inherited/copied verbatim from the base.

The derived Loss adds per-midiseq2-token-TYPE ERROR metrics (type / elapse / pitch / velocity /
channel / nibble; special & sep omitted), aggregated nan-safe via WeightedValue + a custom stat().
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import register_model
from ...utils.weightedValue import WeightedValue
from ...bgpt.decoders import PatchLevelDecoder
from .midiBgptTrans import _build_patch_config, _load_lyl_encoder_weights
from .midiBgptTransSelfAttn import MidiBgptTransSelfAttn
from ..data.seq2CondPachifier import Midiseq2Tokenizer, _HEADER_EVENTS, _FIELD_EVENTS


@register_model
class MidiSeq2BgptSelfAttn (MidiBgptTransSelfAttn):
	'''MidiBgptTransSelfAttn with an independent lilylet encoder width (lyl_patch_size) for the
	dual-width midiseq2 joint layout. Overrides only __init__ (rebuild the encoder) and forward
	(narrow the encoder input slice); all other methods are inherited unchanged.
	'''

	def __init__ (self, lyl_vocab_size=256, midi_vocab_size=582, patch_size=64, lyl_patch_size=16,
		lyl_base_type='llama', lyl_hidden_size=512, lyl_patch_num_layers=8, lyl_patch_length=1024,
		lyl_n_head=8, lyl_intermediate_size=2048, lyl_num_key_value_heads=8,
		lyl_encoder_weights=None, freeze_lyl_encoder=False,
		d_model=768, n_dec_layer=6, token_num_layers=3, n_head=12,
		d_inner=None, dec_num_key_value_heads=None, num_key_value_heads=None, dropout=0.1, **_):
		# Build the base at patch_size (64) — this constructs a throwaway 64-wide lyl encoder we
		# immediately replace below. Suppress the base's (shape-mismatched) weight load + freeze so
		# there is no misleading "loaded 0 tensors" print and nothing is frozen prematurely.
		super().__init__(lyl_vocab_size=lyl_vocab_size, midi_vocab_size=midi_vocab_size,
			patch_size=patch_size, lyl_base_type=lyl_base_type, lyl_hidden_size=lyl_hidden_size,
			lyl_patch_num_layers=lyl_patch_num_layers, lyl_patch_length=lyl_patch_length,
			lyl_n_head=lyl_n_head, lyl_intermediate_size=lyl_intermediate_size,
			lyl_num_key_value_heads=lyl_num_key_value_heads,
			lyl_encoder_weights=None, freeze_lyl_encoder=False,
			d_model=d_model, n_dec_layer=n_dec_layer, token_num_layers=token_num_layers,
			n_head=n_head, d_inner=d_inner, dec_num_key_value_heads=dec_num_key_value_heads,
			num_key_value_heads=num_key_value_heads, dropout=dropout)

		self.lyl_patch_size = lyl_patch_size

		# --- rebuild the lyl encoder at lyl_patch_size (16) so pretrained weights load 1:1 ---
		lyl_config = _build_patch_config(lyl_base_type, lyl_hidden_size, lyl_patch_num_layers,
			lyl_patch_length, lyl_n_head, lyl_intermediate_size, lyl_num_key_value_heads)
		self.lyl_encoder = PatchLevelDecoder(lyl_config, lyl_patch_size, lyl_vocab_size)

		if lyl_encoder_weights and os.path.exists(lyl_encoder_weights):
			n, missing, unexpected = _load_lyl_encoder_weights(self.lyl_encoder, lyl_encoder_weights)
			print(f'[MidiSeq2BgptSelfAttn] loaded {n} lyl-encoder tensors from {lyl_encoder_weights}'
				+ (f' (missing {len(missing)}, unexpected {len(unexpected)})' if missing or unexpected else ''))
		elif lyl_encoder_weights:
			print(f'[MidiSeq2BgptSelfAttn] WARNING: lyl_encoder_weights not found: {lyl_encoder_weights}')

		# apply freeze AFTER the rebuild (base __init__ was called with freeze_lyl_encoder=False).
		self.freeze_lyl_encoder = freeze_lyl_encoder
		if freeze_lyl_encoder:
			for p in self.lyl_encoder.parameters():
				p.requires_grad = False
			self.lyl_encoder.eval()

	def forward (self, patches, masks, modality, attn_mask, target_masks=None,
		positions=None, lyl_counts=None):
		'''Copy of MidiBgptTransSelfAttn.forward diverging ONLY at the lyl-encoder input slice
		(first lyl_patch_size columns instead of the full patch_size). Keep in sync if the base
		forward changes.
		'''
		patches = patches.reshape(len(patches), -1, self.patch_size)
		B, T, _ = patches.shape
		Lmax = int(lyl_counts.max().item()) if lyl_counts is not None else 0

		# --- lyl encoder over the lilylet prefix; feed ONLY the first lyl_patch_size (16) columns ---
		if Lmax > 0:
			# CHANGED vs base: feed only the first lyl_patch_size (16) columns. Also clamp ids to the
			# lyl vocab — in a mixed-length batch the [lyl_count, Lmax) rows of a shorter sample are
			# MIDI patches whose midiseq2 ids (up to 837) exceed lyl_vocab_size, which would blow up
			# the encoder's F.one_hot(num_classes=lyl_vocab). Those rows are masked (lyl_real) and
			# their memory is never scattered (is_lyl False below), so clamping is a safe no-op there.
			lyl_patches = patches[:, :Lmax, :self.lyl_patch_size].clamp(max=self.lyl_vocab_size - 1)
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
		dec = self.decoder(inputs_embeds=embeds, attention_mask=attn_mask,
			position_ids=positions)['last_hidden_state']				# [B,T,d_model]

		if target_masks is None:
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

		# Next-patch prediction: decoded patch i predicts patch i+1.
		left_shift = torch.zeros_like(masks)
		left_shift[:, :-1] = target_masks[:, 1:]
		left_shift = left_shift * masks

		# Degenerate guard (see MidiBgptTrans.forward): rebuild one (context, target) pair.
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


# ---------------------------------------------------------------------------------------
# midiseq2 token-TYPE partition (for grouped error metrics)
# ---------------------------------------------------------------------------------------

# class code -> metric name. Derived from the vocab TOKEN STRINGS (never hardcoded ids), so the
# map tracks assets/midiseq2Vocab.yaml exactly (same asset the feeder consumes). Reported as
# per-type ERROR rate. special (<...>) and sep (_/-) are intentionally NOT reported (structural /
# negligible); their ids still get a class code below but are absent from _TYPE_NAMES so they are
# skipped by the grouped metric loop.
_TYPE_NAMES = {
	1: 'err_type',		# event-type keywords (note_on, set_tempo, ticks_per_beat, ...)
	2: 'err_elapse',	# E… delta run
	3: 'err_channel',	# C… channel
	4: 'err_pitch',		# #… arg3 (pitch / controller type)
	5: 'err_vel',		# $… arg4 (velocity / controller value)
	6: 'err_nibble',	# bare hex nibble of a wide value
}
# class codes assigned but deliberately not reported: 0 = special (<...>), 7 = sep (_/-).

# token-type NAME -> class code, for config-facing loss weights (loss_type_weights below). Names
# match the _TYPE_NAMES stems (drop the 'err_' prefix) plus 'special'/'sep' so every class can be
# weighted. Unknown names in a config raise (typo guard).
_TYPE_CODES = {
	'special': 0,
	'type': 1,		# event-type keywords (the "prefix" token of each event)
	'elapse': 2,
	'channel': 3,
	'pitch': 4,
	'vel': 5,
	'nibble': 6,
	'sep': 7,
}


def _build_type_map (tokenizer):
	'''Classify each vocab id by its token string -> a [vocab] long tensor of class codes.'''
	codes = torch.zeros(len(tokenizer.tokens), dtype=torch.long)
	for i, tok in enumerate(tokenizer.tokens):
		if tok.startswith('<') and tok.endswith('>'):
			c = 0
		elif tok in _HEADER_EVENTS or tok in _FIELD_EVENTS:
			c = 1
		elif tokenizer._is_elapse(tok):
			c = 2
		elif tok.startswith('C'):
			c = 3
		elif tok.startswith('#'):
			c = 4
		elif tok.startswith('$'):
			c = 5
		elif len(tok) == 1 and tok in '0123456789abcdef':
			c = 6
		else:
			c = 7		# '_' / '-' (and any residual)
		codes[i] = c
	return codes


@register_model
class MidiSeq2BgptSelfAttnLoss (nn.Module):
	'''Training wrapper for MidiSeq2BgptSelfAttn: loss + per-token-type error metrics.

	Reuses the base's token-accuracy convention (BOS-prepend, causal shift, <pad>->-100). The
	grouped metrics are WeightedValue(correct_rate, count) per token type so cross-batch averaging
	is nan-safe (an empty class contributes weight 0 and is skipped) — see stat().
	'''

	def __init__ (self, loss_type_weights=None, **kw_args):
		'''loss_type_weights: optional {token-type-name: float} overriding the per-type cross-entropy
		weight (default weight 1 for every type). Names are the _TYPE_CODES keys (special / type /
		elapse / channel / pitch / vel / nibble / sep). When all weights are 1 (the default) the loss
		is byte-identical to the token decoder's built-in uniform cross-entropy, so it is a strict
		generalization. Unknown names raise (typo guard).'''
		super().__init__()
		self.deducer = MidiSeq2BgptSelfAttn(**kw_args)
		# token-type map, derived once from the SAME vocab asset the feeder consumes.
		type_map = _build_type_map(Midiseq2Tokenizer())
		self.register_buffer('type_of_id', type_map, persistent=False)
		self.type_names = dict(_TYPE_NAMES)

		# --- per-vocab-id cross-entropy weight vector, built from the type -> weight config ---
		# start uniform (weight 1 everywhere -> reproduces the built-in CE), then scale each type's
		# ids by its configured weight. weight_of_id[id] is the CE weight of that token id; the -100
		# (pad) label is excluded from the loss regardless, so pad ids' weights never matter.
		self.loss_type_weights = dict(loss_type_weights) if loss_type_weights else {}
		type_weight = torch.ones(len(_TYPE_CODES))				# indexed by class CODE
		for name, w in self.loss_type_weights.items():
			if name not in _TYPE_CODES:
				raise ValueError(f'unknown loss_type_weights key {name!r}; valid: {sorted(_TYPE_CODES)}')
			type_weight[_TYPE_CODES[name]] = float(w)
		weight_of_id = type_weight[type_map]					# [vocab], gather code-weight per id
		self.register_buffer('ce_weight_of_id', weight_of_id, persistent=False)
		# only build a weighted CE path when some weight actually deviates from 1 (else reuse the
		# decoder's own loss for exact parity + zero overhead).
		self.weighted_loss = any(float(w) != 1.0 for w in self.loss_type_weights.values())

	def training_parameters (self):
		return self.deducer.parameters_trainable() + list(self.deducer.buffers())

	def validation_parameters (self):
		return []

	def _weighted_loss (self, output, target_patches):
		'''Per-token-type-weighted cross-entropy over the token-decoder logits.

		Reconstructs the SAME labels the token decoder used (BOS-prepend, causal shift, <pad> -> -100)
		and reweights each position's CE by ce_weight_of_id[target token]. With all weights 1 this
		equals output.loss (HF's default mean CE); with non-unit weights it is a per-token-weighted
		mean, i.e. sum_i w_i * ce_i / sum_i w_i over valid positions — the standard F.cross_entropy
		semantics under a class `weight` vector. Keeps the -100 padding exclusion intact.
		'''
		token_targets = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.deducer.bos_token_id, target_patches), dim=1)
		labels = token_targets.masked_fill(token_targets == self.deducer.special_token_id, -100)
		shift_logits = output.logits[:, :-1, :].contiguous()			# [N, patch_size, vocab]
		shift_labels = labels[:, 1:].contiguous()						# [N, patch_size]
		return F.cross_entropy(
			shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
			weight=self.ce_weight_of_id.to(shift_logits.dtype), ignore_index=-100)

	def _token_accuracy (self, output, target_patches):
		'''Next-token accuracy over valid (non-pad) midi token positions (same as the base).'''
		token_targets = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.deducer.bos_token_id, target_patches), dim=1)
		labels = token_targets.masked_fill(token_targets == self.deducer.special_token_id, -100)
		shift_logits = output.logits[:, :-1, :]
		shift_labels = labels[:, 1:]
		valid = shift_labels != -100
		if not valid.any():
			return 0.0
		return (shift_logits.argmax(dim=-1)[valid] == shift_labels[valid]).float().mean().item()

	def _grouped_error (self, output, target_patches):
		'''Per-token-type next-token ERROR rate -> {name: WeightedValue(err_rate, count)}.

		Groups each valid target position by the TARGET token's type (what should be emitted),
		using the same shifted-labels convention as _token_accuracy. special/sep types are not in
		self.type_names, so they are skipped.
		'''
		token_targets = torch.cat(
			(torch.ones_like(target_patches[:, 0:1]) * self.deducer.bos_token_id, target_patches), dim=1)
		labels = token_targets.masked_fill(token_targets == self.deducer.special_token_id, -100)
		shift_labels = labels[:, 1:]									# [N, patch_size]
		preds = output.logits[:, :-1, :].argmax(dim=-1)				# [N, patch_size]
		valid = shift_labels != -100
		correct = (preds == shift_labels) & valid
		type_of = self.type_of_id.to(shift_labels.device)
		tgt_class = type_of[shift_labels.clamp(min=0)]				# [N, patch_size] (-100 slots masked out below)

		out = {}
		for code, name in self.type_names.items():
			sel = valid & (tgt_class == code)
			cnt = int(sel.sum())
			err = (1.0 - correct[sel].float().mean().item()) if cnt > 0 else 0.0
			out[name] = WeightedValue.from_value(err, cnt)			# weight 0 when the class is absent
		return out

	def stat (self, metric_data, n_batch):
		'''Aggregate accumulated metrics. Headline scalars (acc / err) stay top-level; every
		per-token-type ERROR (the WeightedValue groups) is folded into a single `error`
		dict so the trainer's reportScalars renders them as ONE `error/*` TensorBoard panel instead
		of a panel each — same grouping convention as topology.RectifySieveJointer2Loss.stat's
		`accuracy` dict. Auto-detected by the trainer.'''
		out = {}
		error = {}
		for k, v in metric_data.items():
			if isinstance(v, WeightedValue):
				# skip a class absent from the whole epoch (weight 0 -> value is inf); logging it
				# would pollute the metric stream (e.g. channel is always C0/omitted in this corpus).
				if v.weight != 0:
					error[k[4:] if k.startswith('err_') else k] = v.value	# strip 'err_' -> error/<type>
			else:
				out[k] = v / n_batch
		if error:
			out['error'] = error
		return out

	def forward (self, batch):
		output, target = self.deducer(
			batch['input_patches'], batch['input_masks'], batch['modality'], batch['attn_mask'],
			batch.get('input_targets'), batch.get('input_positions'), batch.get('lyl_counts'))

		# per-token-type-weighted CE when configured; else the decoder's own uniform loss (exact parity).
		loss = self._weighted_loss(output, target) if self.weighted_loss else output.loss

		with torch.no_grad():
			acc = self._token_accuracy(output, target)
			metrics = {'acc': acc, 'err': 1 - acc}
			if not self.training:
				metrics.update(self._grouped_error(output, target))

		return loss, metrics

	def inspectRun (self, batch):
		output, target = self.deducer(
			batch['input_patches'], batch['input_masks'], batch['modality'], batch['attn_mask'],
			batch.get('input_targets'), batch.get('input_positions'), batch.get('lyl_counts'))
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
