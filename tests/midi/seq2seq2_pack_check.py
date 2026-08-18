#!/usr/bin/env python3
'''Checks for `pack: 'split'` (Seq2Seq2) and MidiTranslatorEncDec, the encoder-decoder arm.

Run: python tests/midi/seq2seq2_pack_check.py [--root DIR] [--samples N]

The whole reason the split form exists is to let an encoder-decoder run be COMPARED against the
decoder-only run on the same corpus. That comparison is only meaningful if the two forms ask the model
to predict the same thing, so check 2 is the load-bearing one here: the supervised label set must
match element for element between 'flat' and 'split'. Everything else is plumbing around it.

  1. shapes        — split keys, dtypes, widths; source carries no <sep>; decoder starts with one
  2. label parity  — flat and split select the SAME labels, in the same order, for the same crop
  3. positions     — both halves stay on the pos_style axis; the pad tail continues each run
  4. skip          — start_jitter's unsupervised head lands on the same tokens in both forms
  5. masks         — the encoder mask blocks the pad tail; padded rows stay finite everywhere
  6. pad invariance— a padded batch's real rows match their unpadded selves
  7. cross-attn    — RoPE is absent from cross-attention; the source mask reaches it
  8. loss geometry — logits[:, j] aligns with labels[:, j]; loss/metrics finite; only <sep> leads
  9. generate      — encodes once, stops at <eos>, respects the position axis
 10. checkpoint    — deducer state_dict round-trips; parameters_trainable is flag-independent
 11. overfit       — one batch to acc 1.0, which an off-by-one alignment cannot do (--quick skips it)
'''

import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from starry.midi.data.seq2seq2 import Seq2Seq2						# noqa: E402
from starry.midi.models.midiTranslator import MidiTranslatorLoss	# noqa: E402
from starry.midi.models.midiTranslatorEncDec import (				# noqa: E402
	MidiTranslatorEncDec, MidiTranslatorEncDecLoss, _attn_mask, _keep_mask)
from starry.transformer.sub_layers import MultiHeadAttention	# noqa: E402


DEFAULT_ROOT = '/home/camus/data/midi/test202608'
SOURCE_DIR = 'midi-seq2-score'
TARGET_DIR = 'midi-seq2-irregular'

PASS, FAIL = [], []


def report (name, ok, detail=''):
	(PASS if ok else FAIL).append(name)
	print(f'{"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}')
	return ok


def feeder (root, pack='flat', split='*0/1', **args):
	'''One deterministic feeder. random_crop=False makes every crop a function of its index, so a
	'flat' and a 'split' feeder built the same way see the SAME crops — which is what lets check 2
	compare them directly instead of having to thread one feeder's decisions into the other.'''
	options = dict(source_dir=SOURCE_DIR, target_dir=TARGET_DIR, random_crop=False, seed=0,
		pack=pack, **args)
	(dataset,) = Seq2Seq2.load(root, options, splits=split)
	return dataset


def check_shapes (root, samples):
	flat = feeder(root)
	split = feeder(root, pack='split')
	batch = split.collateBatch([split[i] for i in range(samples)])

	expected = {'source_ids', 'source_masks', 'source_position_ids', 'decoder_input_ids',
		'decoder_masks', 'decoder_position_ids', 'labels', 'target_mask'}
	report('1a keys', set(batch) == expected, f'{sorted(set(batch) ^ expected) or "exact"}')
	report('1b dtypes', all(v.dtype == torch.long for v in batch.values()))

	b, s = batch['source_ids'].shape
	u = batch['decoder_input_ids'].shape[1]
	report('1c widths', b == samples
		and batch['source_masks'].shape == (b, s)
		and batch['source_position_ids'].shape == (b, s)
		and batch['decoder_masks'].shape == (b, u)
		and batch['decoder_position_ids'].shape == (b, u)
		and batch['labels'].shape == (b, u)
		and batch['target_mask'].shape == (b, u), f'B={b} S={s} U={u}')

	sep = split.tokenizer.sep_id
	# The source half must not carry <sep>: it is the decoder's start token, and a copy on the encoder
	# side would make the boundary ambiguous to any tool reading the batch back.
	report('1d no sep in source', not bool((batch['source_ids'] == sep).any()))
	report('1e decoder starts at sep', bool((batch['decoder_input_ids'][:, 0] == sep).all()))
	# Every row's decoder input holds exactly one <sep>, at position 0.
	report('1f one sep per row', bool((batch['decoder_input_ids'] == sep).sum(dim=1).eq(1).all()))

	# The flat form's own contract is unchanged by the refactor.
	flat_batch = flat.collateBatch([flat[i] for i in range(samples)])
	report('1g flat keys intact',
		set(flat_batch) == {'input_ids', 'masks', 'target_mask', 'sep_index', 'position_ids'})
	return flat, split


def check_label_parity (root, samples):
	'''THE check: for the same crop, both forms supervise the same labels in the same order.

	Built by running each form's own loss `_shift` over its own batch, so this compares the tensors the
	MODELS actually see rather than re-deriving them here — a bug in either _shift shows up.
	'''
	flat = feeder(root)
	split = feeder(root, pack='split')
	indices = list(range(samples))

	flat_batch = flat.collateBatch([flat[i] for i in indices])
	split_batch = split.collateBatch([split[i] for i in indices])

	# _shift needs logits only for their shape, so zeros of the right shape suffice.
	vocab = flat.tokenizer.vocab_size
	flat_loss = MidiTranslatorLoss(d_model=32, n_layer=1, n_head=4, vocab_size=vocab)
	split_loss = MidiTranslatorEncDecLoss(d_model=32, n_layer=1, n_head=4, vocab_size=vocab)

	flat_logits = torch.zeros(*flat_batch['input_ids'].shape, vocab)
	split_logits = torch.zeros(*split_batch['labels'].shape, vocab)
	_, flat_labels = flat_loss._shift(flat_batch, flat_logits)
	_, split_labels = split_loss._shift(split_batch, split_logits)

	report('2a same label count', flat_labels.shape == split_labels.shape,
		f'{tuple(flat_labels.shape)} vs {tuple(split_labels.shape)}')
	same = flat_labels.shape == split_labels.shape and bool((flat_labels == split_labels).all())
	report('2b same labels, same order', same, f'{int(flat_labels.numel())} tokens')

	# Per row too, so a same-multiset-different-rows bug cannot pass 2b by accident.
	flat_counts = flat_batch['target_mask'][:, 1:].sum(dim=1).tolist()
	split_counts = split_batch['target_mask'].sum(dim=1).tolist()
	report('2c same per-row counts', flat_counts == split_counts, f'{flat_counts[:4]}...')

	# And the labels really are the target half: every row ends its supervised run with <eos>.
	eos = split.tokenizer.eos_id
	ends = [int(split_batch['labels'][r, int(n) - 1]) for r, n in enumerate(split_counts)]
	report('2d every target ends with eos', all(e == eos for e in ends))
	return flat_batch, split_batch


def check_positions (root, samples):
	ok = True
	for style in ('flat', 'sep', 'absolute'):
		split = feeder(root, pack='split', pos_style=style)
		flat = feeder(root, pos_style=style)
		batch = split.collateBatch([split[i] for i in range(samples)])
		flat_batch = flat.collateBatch([flat[i] for i in range(samples)])

		# The two halves are cut from ONE position run, so concatenating them back must reproduce the
		# flat form's positions over the same span. Checked row-wise against each row's true lengths,
		# since the pad tails differ between the two shapes.
		joined = True
		for row in range(samples):
			s_len = int(batch['source_masks'][row].sum())
			u_len = int(batch['decoder_masks'][row].sum())
			mine = torch.cat([batch['source_position_ids'][row, :s_len],
				batch['decoder_position_ids'][row, :u_len]])
			theirs = flat_batch['position_ids'][row, :s_len + u_len]
			joined = joined and bool((mine == theirs).all())
		ok = report(f'3a {style}: halves rejoin the flat axis', joined) and ok

		# The pad tail continues each run rather than resetting, in both tensors.
		continued = True
		for key, mask_key in (('source_position_ids', 'source_masks'), ('decoder_position_ids', 'decoder_masks')):
			for row in range(samples):
				length = int(batch[mask_key][row].sum())
				tail = batch[key][row, length:]
				if len(tail):
					step = batch[key][row, length] - batch[key][row, length - 1]
					continued = continued and int(step) == 1 and bool((tail.diff() == 1).all())
		ok = report(f'3b {style}: pad tail continues the run', continued) and ok

		if style in ('sep', 'absolute'):
			# <sep> anchors the decoder at -1 under both non-flat styles.
			ok = report(f'3c {style}: sep at -1',
				bool((batch['decoder_position_ids'][:, 0] == -1).all())) and ok
	return ok


def check_skip (root):
	'''start_jitter leaves a head of the target unsupervised. Both forms must drop the SAME tokens —
	an off-by-one here would silently train the split form on a token the flat form withholds.'''
	kw = dict(mark_mode='tick', start_jitter=8.0, line_range=[40, 96])
	flat = feeder(root, **kw)
	split = feeder(root, pack='split', **kw)
	indices = list(range(24))
	flat_batch = flat.collateBatch([flat[i] for i in indices])
	split_batch = split.collateBatch([split[i] for i in indices])

	skips = [case[3] for case in (split[i] for i in indices)]
	report('4a jitter produced skips', any(s > 0 for s in skips), f'nonzero: {sum(1 for s in skips if s)}')

	vocab = flat.tokenizer.vocab_size
	flat_loss = MidiTranslatorLoss(d_model=32, n_layer=1, n_head=4, vocab_size=vocab)
	split_loss = MidiTranslatorEncDecLoss(d_model=32, n_layer=1, n_head=4, vocab_size=vocab)
	_, flat_labels = flat_loss._shift(flat_batch, torch.zeros(*flat_batch['input_ids'].shape, vocab))
	_, split_labels = split_loss._shift(split_batch, torch.zeros(*split_batch['labels'].shape, vocab))
	report('4b same labels under jitter',
		flat_labels.shape == split_labels.shape and bool((flat_labels == split_labels).all()),
		f'{int(split_labels.numel())} tokens')
	# The skipped head is masked, not absent: labels still hold it, target_mask does not select it.
	held = all(int(split_batch['target_mask'][r, :s].sum()) == 0 for r, s in enumerate(skips) if s)
	report('4c skipped head masked not dropped', held)


def check_masks ():
	'''The mask asymmetry from `_attn_mask`, asserted rather than trusted.'''
	dtype = torch.float32
	pad_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

	# Bidirectional + key padding: real queries must not reach the pad tail.
	enc = _attn_mask(4, pad_mask, False, dtype, torch.device('cpu'))
	blocked = bool((enc[0, 0, :, 2:] < -1e30).all()) and bool((enc[0, 0, :, :2] == 0).all())
	report('5a encoder mask blocks pad keys', blocked)
	report('5b encoder mask keeps full rows open', bool((enc[1, 0] == 0).all()))

	# Causal alone: no row is fully masked, so no softmax over all -inf.
	dec = _attn_mask(4, None, True, dtype, torch.device('cpu'))
	upper = torch.ones(4, 4, dtype=torch.bool).triu(1)
	report('5c causal mask is lower-triangular',
		bool((dec[0, 0][upper] < -1e30).all()) and bool((dec[0, 0][~upper] == 0).all()))
	report('5d no fully-masked row in causal mask',
		bool((dec[0, 0] > -1e30).any(dim=-1).all()))

	report('5e no mask when nothing to mask', _attn_mask(4, None, False, dtype, torch.device('cpu')) is None)

	# The cross mask is a different KIND: a boolean keep-mask for the repo's MultiHeadAttention, which
	# unsqueezes its own head axis. [B, 1, Lk] must therefore broadcast over heads and query positions.
	keep = _keep_mask(pad_mask)
	report('5h keep-mask shape', tuple(keep.shape) == (2, 1, 4) and keep.dtype == torch.bool,
		f'{tuple(keep.shape)} {keep.dtype}')
	report('5i keep-mask marks real tokens', keep[0, 0].tolist() == [True, True, False, False])
	report('5j no keep-mask without padding', _keep_mask(None) is None)

	# The decoder's key-padding mask is omitted as REDUNDANT, not to dodge a nan: the fill is a finite
	# finfo.min and the causal block keeps the diagonal at 0, so no query row can lose every key (the
	# validation notebook sweeps every layout). This just pins that the model as wired stays finite.
	model = MidiTranslatorEncDec(vocab_size=64, d_model=32, n_layer=2, n_head=4)
	source = torch.tensor([[7, 8, 9, 0], [7, 8, 9, 10]])
	source_masks = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
	decoder = torch.tensor([[5, 11, 0, 0], [5, 11, 12, 13]])
	logits = model(source, decoder, source_masks)
	report('5f no nan with padded rows', bool(torch.isfinite(logits).all()))
	# And the padded row's REAL positions are finite specifically (the case the nan would poison).
	report('5g padded row real positions finite', bool(torch.isfinite(logits[0, :2]).all()))


def check_pad_invariance ():
	'''A row's real positions must not depend on how much padding shares its batch.

	This is what the continued position run and the encoder mask are FOR, so it is checked on the
	model's output rather than on the tensors.
	'''
	torch.manual_seed(0)
	model = MidiTranslatorEncDec(vocab_size=64, d_model=32, n_layer=2, n_head=4).eval()

	source = torch.tensor([[7, 8, 9]])
	decoder = torch.tensor([[5, 11]])
	source_pos = torch.tensor([[-4, -3, -2]])
	decoder_pos = torch.tensor([[-1, 0]])
	with torch.no_grad():
		alone = model(source, decoder, torch.ones_like(source), source_pos, decoder_pos)

		padded_source = torch.tensor([[7, 8, 9, 0, 0]])
		padded_masks = torch.tensor([[1, 1, 1, 0, 0]])
		padded_source_pos = torch.tensor([[-4, -3, -2, -1, 0]])		# the continued run
		padded_decoder = torch.tensor([[5, 11, 0, 0]])
		padded_decoder_pos = torch.tensor([[-1, 0, 1, 2]])
		together = model(padded_source, padded_decoder, padded_masks, padded_source_pos,
			padded_decoder_pos)

	delta = (alone - together[:, :2]).abs().max().item()
	report('6a padding does not move real positions', delta < 1e-5, f'max |delta| = {delta:.2e}')


def check_cross_attention ():
	model = MidiTranslatorEncDec(vocab_size=64, d_model=32, n_layer=2, n_head=4).eval()
	# Cross-attention is the repo's module and carries no RoPE; self-attention is llama's and does.
	report('7a cross-attention is the repo MultiHeadAttention',
		all(isinstance(layer.cross_attn, MultiHeadAttention) for layer in model.decoder))
	report('7b decoder self-attention has rotary',
		all(layer.self_attn.rotary is not None for layer in model.decoder))
	report('7c encoder self-attention has rotary',
		all(layer.self_attn.rotary is not None for layer in model.encoder))

	# SelfAttention takes no key/value argument at all, so RoPE cannot be applied across the two
	# position axes even by mistake — the absence is structural, not a convention to remember.
	import inspect
	report('7d self-attention cannot take a second sequence',
		'kv' not in inspect.signature(model.decoder[0].self_attn.forward).parameters)

	# The source mask reaches cross-attention: a masked source token must not influence the decoder.
	source = torch.tensor([[7, 8, 9, 10]])
	masks = torch.tensor([[1, 1, 1, 0]])
	decoder = torch.tensor([[5, 11, 12]])
	with torch.no_grad():
		a = model(source, decoder, masks)
		changed = source.clone()
		changed[0, 3] = 42						# perturb only the masked position
		b = model(changed, decoder, masks)
	report('7e masked source token is ignored', bool(torch.allclose(a, b, atol=1e-6)),
		f'max |delta| = {(a - b).abs().max().item():.2e}')

	# And an UNmasked change does move the output, so 7e is not passing because the source is ignored.
	with torch.no_grad():
		other = source.clone()
		other[0, 1] = 42
		c = model(other, decoder, masks)
	report('7f unmasked source token matters', not bool(torch.allclose(a, c, atol=1e-6)),
		f'max |delta| = {(a - c).abs().max().item():.2e}')

	# Cross-attention weights are available for inspection and are a distribution over the source.
	with torch.no_grad():
		_, weights = model.decode(model.encode(source, masks), decoder, masks, need_weights=True)
	rows = weights[0].sum(dim=-1)
	report('7g cross weights sum to 1', bool(torch.allclose(rows, torch.ones_like(rows), atol=1e-5)))
	report('7h cross weights zero on masked source',
		float(weights[0][..., 3].abs().max()) < 1e-6)
	report('7i one weight tensor per decoder layer', len(weights) == len(model.decoder))


def check_loss_geometry (root, samples):
	split = feeder(root, pack='split', pos_style='sep')
	batch = split.collateBatch([split[i] for i in range(samples)])
	model = MidiTranslatorEncDecLoss(d_model=64, n_layer=2, n_head=4,
		vocab_size=split.tokenizer.vocab_size)
	model.train()
	loss, metrics = model(batch)
	report('8a loss finite', bool(torch.isfinite(loss)), f'loss = {loss.item():.4f}')
	report('8b loss backward', (loss.backward() or True)
		and all(p.grad is not None for p in model.deducer.parameters() if p.requires_grad))
	report('8c metrics present', {'acc', 'err'} <= set(metrics))

	model.eval()
	with torch.no_grad():
		_, eval_metrics = model(batch)
	report('8d grouped error on eval', any(k.startswith('err_') for k in eval_metrics))
	stat = model.stat({k: v for k, v in eval_metrics.items()}, 1)
	report('8e stat folds error panel', 'error' in stat and isinstance(stat['error'], dict))

	with torch.no_grad():
		inspected = model.inspectRun(batch)
	report('8f inspectRun works', {'loss', 'acc', 'err', 'logits', 'pred', 'labels', 'n_target'}
		<= set(inspected))

	# The alignment claim itself: logits[:, j] must be a function of decoder inputs up to j only.
	# Perturbing a LATER decoder input must not change an earlier logit (causality), while perturbing
	# an earlier one must change a later logit.
	deducer = model.deducer.eval()
	source, masks = batch['source_ids'][:1], batch['source_masks'][:1]
	decoder = batch['decoder_input_ids'][:1, :6].clone()
	with torch.no_grad():
		base = deducer(source, decoder, masks)
		later = decoder.clone()
		later[0, 5] = (later[0, 5] + 1) % deducer.vocab_size
		after = deducer(source, later, masks)
		earlier = decoder.clone()
		earlier[0, 1] = (earlier[0, 1] + 1) % deducer.vocab_size
		before = deducer(source, earlier, masks)
	report('8g decoder is causal', bool(torch.allclose(base[:, :5], after[:, :5], atol=1e-6)))
	report('8h earlier token moves later logits',
		not bool(torch.allclose(base[:, 3:], before[:, 3:], atol=1e-6)))


def check_generate (root):
	split = feeder(root, pack='split', pos_style='sep')
	batch = split.collateBatch([split[0]])
	model = MidiTranslatorEncDec(vocab_size=split.tokenizer.vocab_size, d_model=64, n_layer=2,
		n_head=4, eos_id=split.tokenizer.eos_id, sep_id=split.tokenizer.sep_id).eval()

	length = int(batch['source_masks'][0].sum())
	source = batch['source_ids'][:1, :length]
	positions = batch['source_position_ids'][:1, :length]
	out = model.generate(source, max_new_tokens=12, source_masks=torch.ones_like(source),
		source_position_ids=positions, decoder_start_position=-1)
	report('9a generate returns ids', out.dim() == 1 and 0 < len(out) <= 12, f'{len(out)} tokens')
	report('9b generate excludes the sep', int(out[0]) != model.sep_id if len(out) else False)

	# An untrained model rarely emits <eos>, so stopping is checked by forcing the stop token.
	forced = model.generate(source, max_new_tokens=12, eos_id=int(out[0]),
		source_masks=torch.ones_like(source), source_position_ids=positions,
		decoder_start_position=-1)
	report('9c stops at eos', int(forced[-1]) == int(out[0]) and len(forced) <= len(out))

	# Batch of one enforced: a silent broadcast over a batched source would produce nonsense.
	failed = False
	try:
		model.generate(batch['source_ids'][:1].expand(2, -1))
	except ValueError:
		failed = True
	report('9d generate rejects a batch', failed)

	# Greedy generation is deterministic, and matches a manual step through decode().
	again = model.generate(source, max_new_tokens=12, source_masks=torch.ones_like(source),
		source_position_ids=positions, decoder_start_position=-1)
	report('9e greedy is deterministic', bool((out == again).all()))

	memory = model.encode(source, torch.ones_like(source), positions)
	with torch.no_grad():
		first = model.decode(memory, torch.tensor([[model.sep_id]]), torch.ones_like(source),
			torch.tensor([[-1]]))[0, -1].argmax()
	report('9f first token matches a manual decode step', int(first) == int(out[0]))


def check_checkpoint (root):
	'''What the trainer relies on: only `.deducer` is saved, and the param list is flag-independent.'''
	vocab = feeder(root, pack='split').tokenizer.vocab_size
	model = MidiTranslatorEncDecLoss(d_model=64, n_layer=2, n_head=4, vocab_size=vocab)
	state = model.deducer.state_dict()
	twin = MidiTranslatorEncDecLoss(d_model=64, n_layer=2, n_head=4, vocab_size=vocab)
	missing, unexpected = twin.deducer.load_state_dict(state, strict=True), None
	report('10a deducer state_dict round-trips', not missing.missing_keys and not missing.unexpected_keys)

	before = len(model.deducer.parameters_trainable())
	model.requires_grad_(False)
	after = len(model.deducer.parameters_trainable())
	report('10b parameters_trainable is flag-independent', before == after and before > 0,
		f'{before} tensors')

	# The type map / CE weights are buffers, not parameters, so they must not be checkpointed.
	report('10c type map not in state_dict',
		not any('type_of_id' in k or 'ce_weight_of_id' in k for k in state))

	# Tied embedding is a real option and must still round-trip.
	tied = MidiTranslatorEncDecLoss(d_model=64, n_layer=2, n_head=4, vocab_size=vocab,
		tie_embedding=True)
	report('10d tie_embedding shares one tensor',
		tied.deducer.lm_head.weight is tied.deducer.embed_tokens.weight)


def check_overfit (root, steps=400):
	'''Overfit ONE batch to acc 1.0. The strongest available check that logits and labels line up.

	An off-by-one between `logits[:, j]` and `labels[:, j]` still trains — it just asks the model to
	predict the token it was handed, or the one after next — and it still produces a finite, falling
	loss. What it cannot do is reach acc 1.0 on a fixed batch, because the required mapping is not a
	function of the visible prefix. So this catches the one class of bug the shape and parity checks
	structurally cannot: a geometry that is self-consistent but shifted.

	Deliberately trained WITHOUT dropout and against a fixed batch, so failing to converge means the
	geometry is wrong rather than the run being unlucky.
	'''
	split = feeder(root, pack='split', pos_style='absolute')
	model = MidiTranslatorEncDecLoss(d_model=192, n_layer=2, n_head=4, dropout=0.0,
		vocab_size=split.tokenizer.vocab_size)
	batch = split.collateBatch([split[i] for i in range(4)])
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

	start = None
	accuracy = 0.0
	for _ in range(steps):
		loss, metrics = model(batch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		start = loss.item() if start is None else start
		accuracy = metrics['acc'].value
	report('11a loss fell', loss.item() < start * 0.2, f'{start:.3f} -> {loss.item():.3f}')
	report('11b overfits one batch', accuracy > 0.99, f'acc {accuracy:.4f} after {steps} steps')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default=DEFAULT_ROOT)
	parser.add_argument('--samples', type=int, default=8)
	parser.add_argument('--quick', action='store_true',
		help='skip check 11 (the overfit run), which is the only slow one')
	args = parser.parse_args()

	torch.manual_seed(0)
	print(f'corpus: {args.root}  {SOURCE_DIR} -> {TARGET_DIR}  samples: {args.samples}\n')

	check_shapes(args.root, args.samples)
	check_label_parity(args.root, args.samples)
	check_positions(args.root, args.samples)
	check_skip(args.root)
	check_masks()
	check_pad_invariance()
	check_cross_attention()
	check_loss_geometry(args.root, args.samples)
	check_generate(args.root)
	check_checkpoint(args.root)
	if not args.quick:
		check_overfit(args.root)

	print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
	if FAIL:
		print('failed: ' + ', '.join(FAIL))
	return 1 if FAIL else 0


if __name__ == '__main__':
	sys.exit(main())
