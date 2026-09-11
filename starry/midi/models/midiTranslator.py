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


class KVDecoder:
	'''Incremental decode over MidiTranslator: prefill the prefix once, then one token per step.

	Exists because the uncached loop is O(T^2) in the WRONG way for this repo's use: it does not
	merely recompute attention, it re-runs the entire stack over the whole prefix for every token.
	A step of the sliding translator generates ~820 tokens off a ~1200-token prefix (measured), so
	the uncached path performs ~820 forwards averaging ~1600 tokens each where one prefill plus 820
	single-token forwards would do.

	Sampling stays OUTSIDE this class. The caller owns temperature / top-k / top-p / the first-token
	<eos> mask, so tools/midi/translateMidiseq2.py keeps its exact selection behaviour and only the
	logits arrive by a faster route.

	TWO POSITION AXES, and conflating them is the one way to get silently wrong output:

	  position_ids   what RoPE rotates by. Comes from the feeder's `pos_style` and may be NEGATIVE
	                 or exceed max_seq_len ('sep' starts the source at a negative offset,
	                 'absolute' carries the token's index in the whole file). Nothing indexes a
	                 table with it.
	  cache_position which cache SLOT a token occupies: always 0..n-1, dense and non-negative. The
	                 backbone builds the causal mask from it, so it must count actual cached entries
	                 and NOT the feeder's positions.

	`masks` is deliberately not accepted. It would have to span the whole cached length rather than
	the new tokens, and generation is single-row with no padding, so there is nothing for it to do.
	'''

	def __init__ (self, model, max_seq_len=None):
		from transformers import DynamicCache

		self.model = model
		self.cache = DynamicCache()
		self.length = 0			# cached entries == next free slot
		if max_seq_len is None:
			max_seq_len = getattr(model, 'max_seq_len', None)
			if max_seq_len is None:
				# Named rather than an AttributeError three frames deep: the usual cause is a stub or a
				# sibling architecture whose forward takes no past_key_values at all (EncDec's decoder
				# is the repo's own blocks). Callers that may hold one should gate on
				# SlidingTranslator.cache_capable instead of catching this.
				raise TypeError(f'{type(model).__name__} has no max_seq_len; KVDecoder needs the '
					'window length, and a model without one probably has no cache support either')
		self.max_seq_len = max_seq_len

	def _tensor (self, values, device):
		return torch.tensor([list(values)], dtype=torch.long, device=device)

	def _trim (self, room):
		'''Drop the OLDEST cache entries so `room` more tokens fit. Returns the slots freed.

		DynamicCache.crop keeps the oldest and drops the newest — the opposite of a sliding window —
		so the keys/values are left-sliced directly. `get_seq_length()` reads the tensors, so it
		follows the slice with no extra bookkeeping.

		This deviates from the uncached path and the deviation is not a bug in either: a cached key at
		slot j holds the hidden state it had when the window still reached further back, while
		recomputation rebuilds that state from a window that now STARTS at j.

		The size of it, from `python3 tests/midi/kv_cache_check.py --trim-probe` (random-init d64/l2,
		max_seq_len 32, prefix 20): max |logit difference| 1.19e-07 over the 12 steps before the first
		trim, 1.30e-01 over the 29 after it, and the greedy argmax differs on 7 of 41 steps. So it
		reaches the EMITTED TOKENS, not just the low bits — a run cannot be reproduced across this
		boundary by flipping the flag.

		Keeping the longer history is the standard streaming behaviour and is if anything the better
		of the two, but it is a DIFFERENT function, so nothing may claim bit-equality once a trim has
		happened. The sliding translator never reaches this branch (its max_token 2048 against
		max_seq_len 4096, and measured runs peak at T=2048); MidiTranslator.generate can.
		'''
		excess = self.length + room - self.max_seq_len
		if excess <= 0:
			return 0
		keep = max(0, self.length - excess)
		for layer in self.cache.layers:
			layer.keys = layer.keys[:, :, self.length - keep:, :].contiguous()
			layer.values = layer.values[:, :, self.length - keep:, :].contiguous()
		self.length = keep
		return excess

	def prefill (self, ids, positions, device=None):
		'''Run the prefix in ONE forward. Returns the last position's logits [vocab].'''
		device = device if device is not None else next(self.model.parameters()).device
		ids = self._tensor(ids, device)
		positions = None if positions is None else self._tensor(positions, device)
		# A prefix longer than the window keeps its TAIL, matching the uncached path's ids[:, -max:].
		if ids.shape[1] > self.max_seq_len:
			ids = ids[:, -self.max_seq_len:]
			positions = None if positions is None else positions[:, -self.max_seq_len:]
		self._trim(ids.shape[1])
		slots = torch.arange(self.length, self.length + ids.shape[1], device=device)
		logits = self.model(ids, None, positions, past_key_values=self.cache,
			cache_position=slots, use_cache=True)
		self.length += ids.shape[1]
		return logits[0, -1, :]

	def step (self, token, position=None, device=None):
		'''Feed ONE token. Returns its logits [vocab], i.e. the prediction for the NEXT token.'''
		device = device if device is not None else next(self.model.parameters()).device
		self._trim(1)
		ids = self._tensor([int(token)], device)
		positions = None if position is None else self._tensor([int(position)], device)
		slots = torch.arange(self.length, self.length + 1, device=device)
		logits = self.model(ids, None, positions, past_key_values=self.cache,
			cache_position=slots, use_cache=True)
		self.length += 1
		return logits[0, -1, :]


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

	def forward (self, input_ids, masks=None, position_ids=None, past_key_values=None,
		cache_position=None, use_cache=False):
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

		The last three arguments are INFERENCE-ONLY and default to the uncached behaviour, so the
		training call (`_logits` passes three positional arguments) is untouched. `input_ids` then
		carries only the NEW tokens and `cache_position` says which cache slots they occupy; see
		KVDecoder for why that is a separate axis from `position_ids`.
		'''
		out = self.backbone(input_ids=input_ids, attention_mask=masks, position_ids=position_ids,
			past_key_values=past_key_values, cache_position=cache_position, use_cache=use_cache)
		return self.lm_head(out.last_hidden_state)

	@torch.no_grad()
	def generate (self, prefix_ids, max_new_tokens=512, eos_id=None, temperature=0.0, masks=None,
		position_ids=None, use_cache=True):
		'''Free-run continuation of ONE prefix (the source half ++ <sep>, optionally ++ <bos>).

		prefix_ids: LongTensor [T] or [1, T]. temperature 0 = greedy, else sample from the softmax.
		Returns the generated ids only (the prefix is NOT included), stopping at eos_id or the
		model's configured target EOS when eos_id is None.

		position_ids: the prefix's positions ([T] or [1, T]), for a feeder using a non-'flat' pos_style.
		Each generated token CONTINUES that run (+1 per step), which is what the target half does under
		every style. Passing None uses the backbone's default 0..T-1, i.e. 'flat'.

		`use_cache` picks the decode route. True (default) prefills the prefix once and then feeds one
		token per step through a KVDecoder, which is what makes bulk decoding affordable. False keeps
		the original recompute-the-whole-prefix loop, retained because it is the reference the cached
		path is checked against (tests/midi/kv_cache_check.py) — not because it is otherwise useful.

		The two agree to float32 noise while the sequence stays inside max_seq_len. Past that they
		diverge for a structural reason, documented in KVDecoder._trim: the cached run keeps hidden
		states computed from a longer history, the recompute run rebuilds them from the cropped
		window. `masks` is not supported with the cache (it would have to span the whole cached
		length, and single-row generation has no padding to mask), so passing one selects the
		uncached path.
		'''
		if use_cache and masks is None:
			return self._generate_cached(prefix_ids, max_new_tokens, eos_id, temperature, position_ids)
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
			# An explicit all-ones mask when the caller gave none: attention_mask=None with no cache
			# makes transformers infer packed-sequence boundaries from position_ids and block attention
			# across each non-unit jump. 'flat' and 'sep' are monotone unit-step and unaffected, but
			# 'absolute' jumps at <sep> and hides the ENTIRE source half from the target. Training
			# always passes masks (Seq2Seq2._collate_flat), so ones is what it computed.
			ones = torch.ones_like(window) if mask_window is None else mask_window
			logits = self.forward(window, ones, pos_window)[:, -1, :]
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


	@torch.no_grad()
	def _generate_cached (self, prefix_ids, max_new_tokens, eos_id, temperature, position_ids):
		'''The KVDecoder route for `generate`. Same selection rule, same return value, O(T) decode.'''
		ids = prefix_ids if prefix_ids.dim() == 2 else prefix_ids.unsqueeze(0)
		if ids.shape[0] != 1:
			raise ValueError('cached generate handles one row at a time')
		pos = None
		if position_ids is not None:
			pos = position_ids if position_ids.dim() == 2 else position_ids.unsqueeze(0)

		decoder = KVDecoder(self)
		logits = decoder.prefill(ids[0].tolist(), None if pos is None else pos[0].tolist(),
			device=ids.device)
		next_pos = None if pos is None else int(pos[0, -1]) + 1

		out = []
		stop = self.eos_id if eos_id is None else eos_id
		for _ in range(max_new_tokens):
			if temperature and temperature > 0:
				nxt = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item())
			else:
				nxt = int(logits.argmax().item())
			out.append(nxt)
			if nxt == stop:
				break
			logits = decoder.step(nxt, next_pos, device=ids.device)
			if next_pos is not None:
				next_pos += 1
		return torch.tensor(out, dtype=torch.long, device=ids.device)


@register_model
class MidiTranslatorLoss (nn.Module):
	'''Training wrapper: masked next-token cross-entropy + per-token-type error metrics.

	Named `MidiTranslator` + 'Loss' because both trainers call
	`loadModel(config['model'], postfix='Loss')` — the config says `type: MidiTranslator` and this
	is what gets constructed, with `config.model.args` as its kwargs. Only `self.deducer` is
	checkpointed, so every trainable tensor lives inside it and the two type maps here are
	`persistent=False` buffers (derived from the vocab asset, never learned).

	`DEDUCER` names the module this wraps. It is a class attribute so a sibling architecture can
	subclass this wrapper and inherit the parts that MUST NOT differ between architectures — the
	vocabulary resolution, the type map, the per-type CE weights, the metrics and their aggregation —
	while overriding only the three things that are genuinely architecture-specific (`DEDUCER`,
	`_logits`, `_shift`). See midiTranslatorEncDec.MidiTranslatorEncDecLoss.

	The per-type metrics reuse midiSeq2BgptSelfAttn's vocab partition (`_build_type_map`), which
	classifies ids by TOKEN STRING against the same `assets/midiseq2Vocab.yaml` the feeder reads —
	but the accuracy/error code is written fresh for the flat `target_mask` form rather than
	adapted, since the sibling's version reconstructs patch labels via a BOS-prepend.
	'''

	DEDUCER = MidiTranslator

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
			self.sep_id = tokenizer.sep_id
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
			self.sep_id = tokenizer.sep_id
			type_map = _build_type_map(tokenizer)
		self.deducer = self.DEDUCER(**kw_args)
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

	def _logits (self, batch):
		'''Batch -> logits. The only place that knows the deducer's call signature, so `forward` and
		`inspectRun` cannot drift apart from each other or from a subclass's architecture.'''
		return self.deducer(batch['input_ids'], batch['masks'], batch.get('position_ids'))

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
		logits = self._logits(batch)
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
		logits = self._logits(batch)
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
