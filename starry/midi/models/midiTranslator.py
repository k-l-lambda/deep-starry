'''MidiTranslator — a decoder-only translator over the FLAT midiseq2 sequence.

Consumes `starry.midi.data.seq2seq2.Seq2Seq2` directly. That feeder emits one flat id sequence per
sample,

    <bos>? source...  <sep>  <bos>? target... <eos>

so translation needs no encoder/decoder split at all: a single causal stack over a single embedding
table sees the source as context and is supervised only on the target half. This is what separates
this model from its siblings in this package (midiBgptTrans, midiSeq2BgptSelfAttn,
midiSeq2PrefixInTokenLevel) — those are PATCH-level bGPT towers whose batch is `[B, T, patch_size]`
with a lilylet condition attached; here the batch is `[B, T]` token ids plus a `target_mask`, and
there is no patch dimension, no cross-attention and no second vocabulary.

Loss convention (the contract the feeder documents): compare `logits[:, i - 1]` against
`input_ids[:, i]` at every `i` where `target_mask` is 1. `<sep>` is therefore the last context
position before the first supervised token, so no source position is ever a target — the source is
conditioning only, and its likelihood is deliberately NOT part of the objective.

`backbone` selects the causal stack. Only 'llama' is wired up for now; the option exists so a second
one can be added without touching the wrapper, the config schema or the metrics.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel

from ...utils.registry import register_model
from ...utils.weightedValue import WeightedValue
from ..data.seq2CondPachifier import Midiseq2Tokenizer
from ..data.unifiedSeq2Tokenizer import UnifiedSeq2Tokenizer
from .midiSeq2BgptSelfAttn import _TYPE_CODES, _TYPE_NAMES, _build_type_map


# backbone name -> builder. Adding one means adding an entry here plus its config builder; the
# deducer only needs `.last_hidden_state` out of whatever this returns.
def _build_backbone (backbone, vocab_size, d_model, n_layer, n_head, d_inner,
	num_key_value_heads, max_seq_len, dropout):
	'''Build the causal stack. Returns (module, hidden_size).'''
	if backbone == 'llama':
		config = LlamaConfig(
			vocab_size=vocab_size,
			hidden_size=d_model,
			num_hidden_layers=n_layer,
			num_attention_heads=n_head,
			num_key_value_heads=num_key_value_heads or n_head,
			intermediate_size=d_inner or d_model * 4,
			max_position_embeddings=max_seq_len,
			attention_dropout=dropout,
		)
		return LlamaModel(config), d_model
	raise ValueError(f'unknown backbone {backbone!r}; supported: llama')


@register_model
class MidiTranslator (nn.Module):
	'''Decoder-only stack over the flat midiseq2 sequence: ids [B, T] -> logits [B, T, vocab].

	The whole model is one embedding table (owned by the backbone), one causal stack, one output
	head. Position 0 has no prediction target of its own; `logits[:, i]` predicts position `i + 1`.
	'''

	# vocab_size has no safe default: it must match the vocabulary the run pinned, and MidiTranslatorLoss
	# always derives it from that file. 582 is only the current asset's size, kept so a bare
	# MidiTranslator() still builds for probes; a real run never relies on it.
	def __init__ (self, vocab_size=582, backbone='llama', d_model=512, n_layer=8, n_head=8,
		d_inner=None, num_key_value_heads=None, max_seq_len=4096, dropout=0.1,
		tie_embedding=False, eos_id=2, **_):
		super().__init__()

		self.vocab_size = vocab_size
		self.backbone_type = backbone
		self.max_seq_len = max_seq_len
		self.eos_id = eos_id

		self.backbone, hidden = _build_backbone(backbone, vocab_size, d_model, n_layer, n_head,
			d_inner, num_key_value_heads, max_seq_len, dropout)
		self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
		if tie_embedding:
			# share the input embedding with the output head. Weight-TYING, so the checkpoint holds
			# one tensor twice by reference — deducer.state_dict() still round-trips.
			self.lm_head.weight = self.backbone.embed_tokens.weight
		else:
			nn.init.normal_(self.lm_head.weight, std=0.02)

	def parameters_trainable (self):
		'''Every parameter — this model has no frozen submodule, so there is nothing to exclude.

		Deliberately NOT filtered on the live `requires_grad` flag. The distributed validator calls
		`model.requires_grad_(False)` (trainerQuantitative.py) before the per-epoch param broadcast,
		so a flag-based filter returns an empty list on the validator while the trainer returns all
		77 tensors — broadcastParam then pairs mismatched sizes and gloo aborts the process
		("op.preamble.length <= op.nbytes. 1716224 vs 128"). Enumerating all parameters yields the
		SAME list in the SAME order on both ranks. See MidiBgptTrans.parameters_trainable.
		'''
		return list(self.parameters())

	def forward (self, input_ids, masks=None, position_ids=None):
		'''
		input_ids:    LongTensor [B, T]
		masks:        LongTensor [B, T] or None — 1 = real token. Padding is right-side only, and the
		              stack is causal, so a padded tail cannot leak into a real position either way;
		              the mask is still passed so padding contributes nothing to the attention softmax.
		position_ids: LongTensor [B, T] or None — explicit RoPE positions from the feeder's `pos_style`.
		              None lets the backbone use its default 0..T-1, which is exactly `pos_style: flat`.
		              Values may be NEGATIVE and may exceed max_seq_len: RoPE computes sin/cos from the
		              value itself, so nothing indexes a table and neither case is out of range.
		Returns: FloatTensor [B, T, vocab] — logits[:, i] predicts position i + 1.
		'''
		out = self.backbone(input_ids=input_ids, attention_mask=masks, position_ids=position_ids)
		return self.lm_head(out.last_hidden_state)

	@torch.no_grad()
	def generate (self, prefix_ids, max_new_tokens=512, eos_id=None, temperature=0.0, masks=None,
		position_ids=None):
		'''Free-run continuation of ONE prefix (the source half ++ <sep>, optionally ++ <bos>).

		prefix_ids: LongTensor [T] or [1, T]. temperature 0 = greedy, else sample from the softmax.
		Returns the generated ids only (the prefix is NOT included), stopping at eos_id or the
		model's configured target EOS when eos_id is None.

		position_ids: the prefix's positions ([T] or [1, T]), for a feeder using a non-'flat' pos_style.
		Each generated token CONTINUES that run (+1 per step), which is what the target half does under
		every style. Passing None uses the backbone's default 0..T-1, i.e. 'flat'.

		No KV cache: this recomputes the whole prefix every step, which is O(T^2) per token and is
		meant for notebook-scale inspection, not for bulk decoding.
		'''
		ids = prefix_ids if prefix_ids.dim() == 2 else prefix_ids.unsqueeze(0)
		pos = None
		if position_ids is not None:
			pos = position_ids if position_ids.dim() == 2 else position_ids.unsqueeze(0)
		mask = None
		if masks is not None:
			mask = masks if masks.dim() == 2 else masks.unsqueeze(0)
			if mask.shape[1] != ids.shape[1]:
				raise ValueError('masks must have the same sequence length as prefix_ids')
		out = []
		for _ in range(max_new_tokens):
			window = ids[:, -self.max_seq_len:]
			pos_window = pos[:, -self.max_seq_len:] if pos is not None else None
			mask_window = mask[:, -self.max_seq_len:] if mask is not None else None
			logits = self.forward(window, mask_window, pos_window)[:, -1, :]
			if temperature and temperature > 0:
				nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
			else:
				nxt = logits.argmax(dim=-1, keepdim=True)
			token = int(nxt.item())
			out.append(token)
			ids = torch.cat((ids, nxt), dim=1)
			if mask is not None:
				mask = torch.cat((mask, torch.ones_like(nxt)), dim=1)
			if pos is not None:
				pos = torch.cat((pos, pos[:, -1:] + 1), dim=1)
			if token == (self.eos_id if eos_id is None else eos_id):
				break
		return torch.tensor(out, dtype=torch.long, device=ids.device)


@register_model
class MidiTranslatorLoss (nn.Module):
	'''Training wrapper: masked next-token cross-entropy + per-token-type error metrics.

	Named `MidiTranslator` + 'Loss' because both trainers call
	`loadModel(config['model'], postfix='Loss')` — the config says `type: MidiTranslator` and this
	is what gets constructed, with `config.model.args` as its kwargs. Only `self.deducer` is
	checkpointed, so every trainable tensor lives inside it and the two type maps here are
	`persistent=False` buffers (derived from the vocab asset, never learned).

	The per-type metrics reuse midiSeq2BgptSelfAttn's vocab partition (`_build_type_map`), which
	classifies ids by TOKEN STRING against the same `assets/midiseq2Vocab.yaml` the feeder reads —
	but the accuracy/error code is written fresh for the flat `target_mask` form rather than
	adapted, since the sibling's version reconstructs patch labels via a BOS-prepend.
	'''

	def __init__ (self, loss_type_weights=None, vocab_path=None, **kw_args):
		'''loss_type_weights: optional {token-type-name: float} per-type cross-entropy weight
		(default 1 for every type, which is plain uniform CE). Names are the `_TYPE_CODES` keys
		(special / type / elapse / channel / pitch / vel / nibble / sep); an unknown name raises.'''
		super().__init__()

		unified = bool(vocab_path and UnifiedSeq2Tokenizer.matches(vocab_path))
		if unified:
			tokenizer = UnifiedSeq2Tokenizer(vocab_path)
			if 'vocab_size' in kw_args and int(kw_args['vocab_size']) != tokenizer.vocab_size:
				raise ValueError(f'unified vocab_size must be {tokenizer.vocab_size}, got {kw_args["vocab_size"]}')
			kw_args['vocab_size'] = tokenizer.vocab_size
			self.pad_id = tokenizer.pad_id
			# One class per unified REGION. The merged layout has three: shared controls, Lilylet
			# content, midiseq2 content — so nothing here may assume the MIDI block still carries its
			# own controls or a fixed number of source rows.
			midi_block = tokenizer.blocks['midiseq2']
			lyl_block = tokenizer.blocks['lilylet']
			midi_local = _build_type_map(Midiseq2Tokenizer())[
				midi_block['local_start']:midi_block['local_start'] + midi_block['size']]
			type_map = torch.full((tokenizer.vocab_size,), 8, dtype=torch.long)		# 8 = lyl
			# Shared controls are modality-neutral; <sep> keeps its own legacy class.
			type_map[:lyl_block['offset']] = _TYPE_CODES['special']
			type_map[tokenizer.sep_id] = _TYPE_CODES['sep']
			type_map[midi_block['offset']:] = midi_local
		else:
			tokenizer = Midiseq2Tokenizer(vocab_path) if vocab_path else Midiseq2Tokenizer()
			kw_args.setdefault('vocab_size', tokenizer.vocab_size)
			self.pad_id = tokenizer.pad_id
			type_map = _build_type_map(tokenizer)
		self.deducer = MidiTranslator(**kw_args)
		self.register_buffer('type_of_id', type_map, persistent=False)
		self.type_names = dict(_TYPE_NAMES)
		valid_type_codes = dict(_TYPE_CODES)
		if isinstance(tokenizer, UnifiedSeq2Tokenizer):
			valid_type_codes['lyl'] = 8
			self.type_names[8] = 'err_lyl'

		# per-vocab-id CE weight, built from the type -> weight config. Uniform (all 1) reproduces
		# plain cross_entropy exactly, so the weighted path is a strict generalization.
		self.loss_type_weights = dict(loss_type_weights) if loss_type_weights else {}
		type_weight = torch.ones(max(valid_type_codes.values()) + 1)		# indexed by class CODE
		for name, w in self.loss_type_weights.items():
			if name not in valid_type_codes:
				raise ValueError(f'unknown loss_type_weights key {name!r}; valid: {sorted(valid_type_codes)}')
			type_weight[valid_type_codes[name]] = float(w)
		self.register_buffer('ce_weight_of_id', type_weight[type_map], persistent=False)
		self.weighted_loss = any(float(w) != 1.0 for w in self.loss_type_weights.values())

	def training_parameters (self):
		return self.deducer.parameters_trainable() + list(self.deducer.buffers())

	def validation_parameters (self):
		return []

	def _shift (self, batch, logits):
		'''The whole loss/metric geometry in one place, so nothing can disagree about the shift.

		Returns (pred_logits, labels) over the SUPERVISED positions only, already flattened:
		position `i` of the target region is predicted by `logits[:, i - 1]`, so the logits are
		dropped by one from the right and the labels by one from the left. `target_mask` excludes
		padding by construction (the feeder sets it only up to each row's true length).
		'''
		labels = batch['input_ids'][:, 1:]						# [B, T-1]
		sel = batch['target_mask'][:, 1:].bool()					# [B, T-1]
		return logits[:, :-1, :][sel], labels[sel]				# [N, vocab], [N]

	def _loss (self, pred, labels):
		if not self.weighted_loss:
			return F.cross_entropy(pred, labels)
		return F.cross_entropy(pred, labels, weight=self.ce_weight_of_id.to(pred.dtype))

	def _grouped_error (self, pred, labels):
		'''Per-token-type next-token ERROR rate -> {name: WeightedValue(err_rate, count)}.

		Grouped by the TARGET token's type (what should have been emitted). A type absent from the
		batch gets weight 0, so cross-batch averaging in stat() stays nan-safe. special/sep are not
		in self.type_names and are therefore not reported.
		'''
		correct = pred.argmax(dim=-1) == labels
		tgt_class = self.type_of_id.to(labels.device)[labels]

		out = {}
		for code, name in self.type_names.items():
			sel = tgt_class == code
			cnt = int(sel.sum())
			err = (1.0 - correct[sel].float().mean().item()) if cnt > 0 else 0.0
			out[name] = WeightedValue.from_value(err, cnt)
		return out

	def stat (self, metric_data, n_batch):
		'''Aggregate accumulated metrics. Every `err_*` WeightedValue folds into ONE `error` dict so
		TensorBoard renders a single `error/*` panel instead of one panel per token type; any other
		WeightedValue keeps its own key at its weighted value; plain scalars are divided by n_batch.
		Auto-detected by the trainer via hasattr.'''
		out, error = {}, {}
		for k, v in metric_data.items():
			if isinstance(v, WeightedValue):
				# weight 0 = the class never appeared this epoch (value is inf); logging it would
				# pollute the metric stream.
				if v.weight != 0:
					(error.__setitem__(k[4:], v.value) if k.startswith('err_')
						else out.__setitem__(k, v.value))
			else:
				out[k] = v / n_batch
		if error:
			out['error'] = error
		return out

	def forward (self, batch):
		logits = self.deducer(batch['input_ids'], batch['masks'], batch.get('position_ids'))
		pred, labels = self._shift(batch, logits)
		loss = self._loss(pred, labels)

		with torch.no_grad():
			n = int(labels.numel())
			acc = (pred.argmax(dim=-1) == labels).float().mean().item() if n else 0.0
			# WeightedValue by supervised-token count: rows differ a lot in target length (the crop
			# is a sampled line range), so a plain per-batch mean would over-weight short targets.
			metrics = {'acc': WeightedValue.from_value(acc, n), 'err': WeightedValue.from_value(1 - acc, n)}
			if not self.training:
				metrics.update(self._grouped_error(pred, labels))

		return loss, metrics

	def inspectRun (self, batch):
		'''Notebook entry point: the metrics plus the raw tensors needed to look at a prediction.'''
		logits = self.deducer(batch['input_ids'], batch['masks'], batch.get('position_ids'))
		pred, labels = self._shift(batch, logits)
		loss = self._loss(pred, labels)
		acc = (pred.argmax(dim=-1) == labels).float().mean().item() if labels.numel() else 0.0
		metrics = {
			'loss': loss.item(),
			'acc': acc,
			'err': 1 - acc,
			'logits': logits,
			'pred': pred.argmax(dim=-1),
			'labels': labels,
			'n_target': int(labels.numel()),
		}
		metrics.update({k: v.value for k, v in self._grouped_error(pred, labels).items()})
		return metrics
