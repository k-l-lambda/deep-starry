'''Checks for tools/midi/translateMidiseq2.py — the sliding-window MidiTranslator inference CLI.

The decisive one is PREFIX PARITY: the script must feed the model exactly what Seq2Seq2 fed it during
training. Everything else in the pipeline can look fine while the prefix quietly differs by a <bos>, a
dropped directive or an off-by-one position, and the only symptom would be poor output that looks like a
model problem. So we build the feeder over the same corpus, take its own crop decision via describe(),
and assert the script reproduces the source half and its positions id-for-id.

The rest are properties of the sliding mechanic itself: that the view advances one measure per step, that
it cannot stall, that rendering round-trips through the vocab, and that encode_lines matches the feeder's
_encode on real files.

Run:  python tests/midi/translate_midiseq2_check.py [--root DIR] [--samples N]
'''

import argparse
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.midi.data.seq2seq2 import Seq2Seq2, _get_file
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer

sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'midi'))
from translateMidiseq2 import (encode_lines, keyword_tokens, positions_for, render_lines,
	plot_attention_step,
	count_note_on, source_header, SlidingTranslator, SlidingEncDecTranslator, is_elapse,
	note_on_events, line_token_offsets, AttentionInspector,
	trim_annotated_prelude, is_elapse_line, is_sustain_line)


DEFAULT_ROOT = os.path.expanduser('~/data/midi/test202608')


def check_encode_parity (root, samples, verbose=False):
	'''encode_lines must agree with the feeder's _encode on real files, both eom settings.'''
	tk = Midiseq2Tokenizer()
	ds = Seq2Seq2(root, '0/1', source_dir='midi-seq2-irregular', target_dir='midi-seq2-score',
		mark_mode='tick', line_range=[20, 256], pos_style='sep', random_crop=False)
	bad = 0
	for index in ds.indices[:samples]:
		case = ds.describe(index)
		src, tgt = case['source'], case['target']
		a, z = case['source_range']
		t0, t1 = case['target_range']
		for lines, eom, label in ((src.lines[a:z], False, 'source'), (tgt.lines[t0:t1], True, 'target')):
			mine = encode_lines(lines, tk, eom)
			theirs, _ = ds._encode(lines, eom, 0)
			if mine != theirs:
				bad += 1
				print(f'  FAIL {case["name"][:8]} {label}: {len(mine)} vs {len(theirs)} ids')
	print(f'{"ok  " if not bad else "FAIL"} encode_lines matches Seq2Seq2._encode '
		f'({samples} files, both halves, {bad} mismatches)')
	return bad == 0


def check_prefix_parity (root, samples, verbose=False):
	'''The decisive test: build_prefix must reproduce the feeder's source half + <sep> exactly.

	We take the feeder's own crop (describe gives the line range it chose), encode that same range
	through the script's path, and compare ids and positions against case['ids'][:sep+1] and
	case['positions'][:sep+1].
	'''
	tk = Midiseq2Tokenizer()
	ds = Seq2Seq2(root, '0/1', source_dir='midi-seq2-irregular', target_dir='midi-seq2-score',
		mark_mode='tick', line_range=[20, 256], pos_style='sep', random_crop=False)
	tr = SlidingTranslator(None, tk, pos_style='sep', src_window=10 ** 9, max_token=10 ** 9)
	bad = 0
	for index in ds.indices[:samples]:
		case = ds.describe(index)
		a, z = case['source_range']
		sep = case['sep']
		src_ids = encode_lines(case['source'].lines[a:z], tk, False)
		# the feeder's head flag is what decides <bos> on both halves
		ids, positions, n_source = tr.build_prefix(src_ids, [], head=case['head'])
		# compare the SOURCE half + <sep>. build_prefix legitimately continues past that with the
		# target half's own <bos> on a head crop (the feeder emits it too, at case['ids'][sep+1]),
		# so it is checked separately rather than being counted as a length mismatch.
		got, want = ids[:n_source + 1], case['ids'][:sep + 1]
		got_pos, want_pos = positions[:n_source + 1], case['positions'][:sep + 1]
		if n_source != sep:
			bad += 1
			print(f'  FAIL {case["name"][:8]} sep index: {n_source} != {sep}')
		elif got != want:
			first = next(i for i, (x, y) in enumerate(zip(got, want)) if x != y)
			bad += 1
			print(f'  FAIL {case["name"][:8]} ids: first diff at {first}')
		elif got_pos != want_pos:
			bad += 1
			print(f'  FAIL {case["name"][:8]} positions: {got_pos[:6]}... vs {want_pos[:6]}...')
		elif case['head'] and ids[n_source + 1:] != [tk.bos_id]:
			bad += 1
			print(f'  FAIL {case["name"][:8]} head crop should prime the target with <bos>')
		elif case['head'] and case['ids'][sep + 1] != tk.bos_id:
			bad += 1
			print(f'  FAIL {case["name"][:8]} feeder disagrees on target <bos>')
		elif verbose:
			print(f'  ok {case["name"][:8]} sep={sep} head={case["head"]} src={len(src_ids)}')
	print(f'{"ok  " if not bad else "FAIL"} prefix parity vs Seq2Seq2.describe '
		f'({samples} files, {bad} mismatches)')
	return bad == 0


def check_encdec_split_inference ():
	'''EncDec prefixes and low-level generation must match the split feeder contract.'''
	import torch
	tk = Midiseq2Tokenizer()
	ok = True
	tr = SlidingEncDecTranslator(None, tk, pos_style='sep', max_token=32)
	ids, pos, n = tr.build_prefix([10, 11], [], head=True)
	want_ids = [tk.bos_id, 10, 11, tk.sep_id, tk.bos_id]
	want_pos = [-4, -3, -2, -1, 0]
	if ids != want_ids or pos != want_pos or n != 3:
		print(f'  FAIL EncDec head split: ids={ids}, pos={pos}, n_source={n}')
		ok = False
	ids2, pos2, n2 = tr.build_prefix([10, 11], [20], head=False)
	if ids2 != [10, 11, tk.sep_id, 20] or pos2 != [-3, -2, -1, 0] or n2 != 2:
		print(f'  FAIL EncDec continuation split: ids={ids2}, pos={pos2}, n_source={n2}')
		ok = False

	class StubEncDec:
		max_seq_len = 64
		encoder = object()
		def __init__ (self):
			self.encode_calls = []
			self.decode_calls = []
		def encode (self, source, masks, positions):
			self.encode_calls.append((source.detach().clone(), masks.detach().clone(), positions.detach().clone()))
			return source.float()
		def decode (self, memory, decoder, source_masks, decoder_positions):
			self.decode_calls.append((decoder.detach().clone(), decoder_positions.detach().clone()))
			v = len(tk.tokens)
			logits = torch.full((1, decoder.shape[1], v), -5.0)
			logits[0, -1, tk.eos_id] = 10.0
			if len(self.decode_calls) == 1:
				logits[0, -1, tk.eom_id] = 9.0
			return logits
	model = StubEncDec()
	tr = SlidingEncDecTranslator(model, tk, pos_style='sep', max_token=32)
	got, forced = tr.generate(want_ids, want_pos, 0.0, 0, 1.0, n_source=3)
	if got != [tk.eom_id] or not forced:
		print(f'  FAIL EncDec EOS-mask generation: ids={got}, forced={forced}')
		ok = False
	if len(model.encode_calls) != 1:
		print(f'  FAIL source encoded {len(model.encode_calls)} times, want once')
		ok = False
	elif model.encode_calls[0][0].tolist() != [[tk.bos_id, 10, 11]] or model.encode_calls[0][2].tolist() != [[-4, -3, -2]]:
		print(f'  FAIL encoded source arguments: {model.encode_calls[0]}')
		ok = False
	elif model.decode_calls[0][0].tolist() != [[tk.sep_id, tk.bos_id]] or model.decode_calls[0][1].tolist() != [[-1, 0]]:
		print(f'  FAIL decoder prefix arguments: {model.decode_calls[0]}')
		ok = False
	print(f'{"ok  " if ok else "FAIL"} EncDec split prefix, one-shot encode, and EOS mask')
	return ok


def check_encdec_inspector_indexing (root):
	'''EncDec inspection must convert joint prefix indices to cross-attention halves exactly.'''
	import torch
	d = os.path.join(root, 'midi-seq2-irregular')
	if not os.path.isdir(d):
		print('skip EncDec inspector indexing: no corpus arm found')
		return True
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	lines = open(os.path.join(d, sorted(os.listdir(d))[0])).read().splitlines()[:100]
	src_ids = encode_lines(lines, tk)
	new_ids = src_ids[:600]

	class StubEncDec:
		encoder = object()
		def __init__ (self, peak):
			self.peak = peak
			self.calls = []
		def encode (self, source, masks, positions):
			return source.float()
		def decode (self, memory, decoder, source_masks, positions, need_weights=False):
			S, U = memory.shape[1], decoder.shape[1]
			# The first layer deliberately peaks elsewhere: inspection must use the final
			# cross-attention layer rather than averaging decoder layers together.
			early = torch.zeros(1, 2, U, S)
			early[:, :, :, (self.peak + 1) % S] = 2.0
			final = torch.zeros(1, 2, U, S)
			final[:, :, :, self.peak] = 1.0
			self.calls.append((decoder.detach().clone(), positions.detach().clone()))
			return torch.zeros(1, U, len(tk.tokens)), (early, final)

	src_events, _, _ = note_on_events(src_ids, tk, kw)
	want_event = src_events[1]
	ok = True
	for head in (True, False):
		# The peak is at one known source-tensor column. On a head window that column includes <bos>.
		model = StubEncDec(want_event['pitch_index'] + int(head))
		insp = AttentionInspector(model, tk, kw, lines, False, 'cpu', query='producer',
			threshold=0.05, top_k=8)
		source = ([tk.bos_id] if head else []) + src_ids
		prefix = source + [tk.sep_id] + ([tk.bos_id] if head else [])
		positions = positions_for('sep', len(source), 1 if head else 0)
		insp.observe(0, prefix, positions, new_ids, src_ids, 0, len(lines), head, 0,
			n_source=len(source))
		if insp.attn_calls != 1 or not insp.links:
			print(f'  FAIL EncDec head={head}: calls={insp.attn_calls}, links={len(insp.links)}')
			ok = False
			continue
		wrong = [link for link in insp.links if link[2] != want_event['pitch_index']]
		if wrong:
			print(f'  FAIL EncDec head={head}: {len(wrong)} link(s) missed peaked source pitch')
			ok = False
	print(f'{"ok  " if ok else "FAIL"} EncDec cross-attention indexing: source <bos> offset and call count')
	return ok


def check_positions ():
	'''pos_style layouts, absolute file-axis positions, and explicit axes.'''
	ok = True
	flat = positions_for('flat', 3, 2)
	if flat != [0, 1, 2, 3, 4, 5]:
		print(f'  FAIL flat: {flat}'); ok = False
	sep = positions_for('sep', 3, 2)
	if sep != [-4, -3, -2, -1, 0, 1]:
		print(f'  FAIL sep: {sep}'); ok = False
	absolute = positions_for('absolute', 3, 2, [4, 5], total_source=10, target_base=7, head=True)
	if absolute != [-8, -7, -6, -1, 6, 7]:
		print(f'  FAIL absolute: {absolute}'); ok = False
	try:
		positions_for('absolute', 3, 2)
		print('  FAIL absolute without axes should raise'); ok = False
	except ValueError:
		pass
	print(f'{"ok  " if ok else "FAIL"} positions_for: flat / sep / absolute layouts and explicit axes')
	return ok


def check_absolute_prefix ():
	'''Absolute prefixes preserve source file coordinates and target primer coordinates.'''
	import torch
	tk = Midiseq2Tokenizer()
	ok = True
	tr = SlidingTranslator(None, tk, pos_style='absolute', max_token=64)
	ids, pos, n = tr.build_prefix([10, 11], [20, 21], head=False,
		source_base=4, total_source=10, target_base=6)
	if ids != [10, 11, tk.sep_id, 20, 21] or pos != [-7, -6, -1, 6, 7] or n != 2:
		print(f'  FAIL absolute continuation prefix: ids={ids}, pos={pos}, n={n}')
		ok = False
	ids, pos, n = tr.build_prefix([10, 11], [], head=True,
		source_base=0, total_source=10, target_base=0)
	if pos != [-12, -11, -10, -1, -1]:
		print(f'  FAIL absolute head prefix: pos={pos}')
		ok = False
	class PositionModel:
		def __init__ (self):
			self.positions = []
		def __call__ (self, ids, mask, positions):
			self.positions.append(positions.detach().clone())
			v = len(tk.tokens)
			logits = torch.full((1, ids.shape[1], v), -10.0)
			logits[0, -1, tk.eom_id] = 10.0
			logits[0, -1, tk.eos_id] = 0.0
			return logits
	model = PositionModel()
	tr.model = model
	prefix, prefix_pos, n = tr.build_prefix([10], [], head=False,
		source_base=4, total_source=10, target_base=6)
	tr.generate(prefix, prefix_pos, 0.0, 0, 1.0, n_source=n, next_position=6)
	if not model.positions or model.positions[0][0, -1].item() != -1:
		print(f'  FAIL absolute empty-primer prefix query: {model.positions}')
		ok = False
	# The first generated query is the prefix tail (-1); the next query carries explicit target position 6.
	if len(model.positions) < 2 or model.positions[1][0, -1].item() != 6:
		print(f'  FAIL absolute empty-primer seed: {model.positions}')
		ok = False
	print(f'{"ok  " if ok else "FAIL"} absolute prefix and empty-primer position seed')
	return ok


def check_advance ():
	'''advance_output uses target-token thresholds, rounds to eom, and cannot stall.'''
	tk = Midiseq2Tokenizer()
	tr = SlidingTranslator(None, tk)
	ok = True
	eom = tk.eom_id
	# first <eom> at index 3 -> new prime_start is 4
	out = [10, 11, 12, eom, 13, 14, eom, 15]
	got = tr.advance_output(out, 0)
	if got != 4:
		print(f'  FAIL eom advance: {got} != 4'); ok = False
	# from 4, the next <eom> is at 6 -> 7
	got = tr.advance_output(out, 4)
	if got != 7:
		print(f'  FAIL second eom advance: {got} != 7'); ok = False
	# no <eom> in view -> the minimum target-token stride (default 1)
	got = tr.advance_output([10, 11, 12, 13, 14, 15], 0)
	if got != 1:
		print(f'  FAIL fallback: {got} != 1 (default token stride)'); ok = False
	# the threshold is relative to the current target view, not the whole stream
	got = tr.advance_output([1, 2, 3, 4, 10, 11, 12, 13], 4)
	if got != 5:
		print(f'  FAIL fallback denominator: {got} != 5 (4 + one token)'); ok = False
	# a single-token view must still advance
	got = tr.advance_output([10], 0)
	if got != 1:
		print(f'  FAIL single-token advance: {got} != 1'); ok = False
	# an empty view cannot advance, and must not go backwards
	got = tr.advance_output([10, 11], 2)
	if got != 2:
		print(f'  FAIL empty view: {got} != 2'); ok = False
	# The public stride is a minimum target-token distance: the cut rounds upward to the next <eom>.
	for tokens, want in ((1, 4), (3, 4), (4, 7), (5, 7)):
		wide = SlidingTranslator(None, tk, advance_tokens=tokens, prime_window=10 ** 6)
		got = wide.advance_output(out, 0)
		if got != want:
			print(f'  FAIL advance_tokens={tokens}: {got} != {want}'); ok = False
	# A threshold beyond the stream clamps at EOF; no boundary after a threshold moves by tokens exactly.
	wide = SlidingTranslator(None, tk, advance_tokens=99, prime_window=10 ** 6)
	if wide.advance_output(out, 0) != len(out):
		print(f'  FAIL oversized token advance: {wide.advance_output(out, 0)} != {len(out)}'); ok = False
	wide = SlidingTranslator(None, tk, advance_tokens=3, prime_window=10 ** 6)
	if wide.advance_output([10, 11, 12, 13, 14], 0) != 3:
		print('  FAIL no-eom token advance should not use half-window fallback'); ok = False
	# The safety ceiling is separate and is applied only by trim_prime afterward.
	if SlidingTranslator(None, tk, advance_tokens=4, prime_window=5).advance_output(out, 0) != 7:
		print('  FAIL stride should be decided before safety trim'); ok = False
	print(f'{"ok  " if ok else "FAIL"} advance_output: token threshold rounded to eom, fallback, no stall')
	return ok


def check_source_advance (root):
	'''advance_source_by_onsets lands after the Nth note_on and is monotone.'''
	name = sorted(os.listdir(os.path.join(root, 'midi-seq2-irregular')))[0]
	lines = open(os.path.join(root, 'midi-seq2-irregular', name)).read().splitlines()
	tk = Midiseq2Tokenizer()
	tr = SlidingTranslator(None, tk)
	ok = True
	cursor = tr.advance_source_by_onsets(lines, 0, 5)
	before = sum(1 for l in lines[:cursor] if l.startswith('note_on'))
	if before != 5:
		print(f'  FAIL 5 onsets consumed {before}'); ok = False
	if not lines[cursor - 1].startswith('note_on'):
		print(f'  FAIL cursor should sit just after a note_on, got {lines[cursor-1]!r}'); ok = False
	if tr.advance_source_by_onsets(lines, 10, 0) != 10:
		print('  FAIL zero onsets should not move'); ok = False
	if tr.advance_source_by_onsets(lines, 0, 10 ** 9) != len(lines):
		print('  FAIL overshoot should clamp to EOF'); ok = False
	print(f'{"ok  " if ok else "FAIL"} advance_source_by_onsets: lands after Nth note_on, clamps at EOF')
	return ok


def check_source_annotation (root):
	"""annotate_source writes bars onto the source that conserve its notes and match the target's count.

	The property that matters is CONSERVATION: every source note_on must land in exactly one bar, so the
	annotated file is the same music with a coordinate added. A boundary rule that lost or duplicated a
	note would still produce a plausible-looking file, and the figure built on it would then be a figure
	of a source that does not exist.

	Driven by a stub `_align_first_src` rather than a real run: the point is the boundary arithmetic, and
	the cases worth covering (a backward match, a bar that matched nothing, a trim) are ones a given
	checkpoint may or may not happen to produce.
	"""
	tk = Midiseq2Tokenizer()
	ok = True
	n = 40
	lines = [f'note_on C1 #{40 + i} $40' for i in range(n)]

	def annotate (first, matched, kept=None, distinct=None):
		tr = SlidingTranslator(None, tk)
		# per bar: EVERY source index its notes matched. A bare int is accepted as a one-note bar so the
		# simple cases stay readable; a list is what the real walk collects.
		tr._align_first_src = [[] if v is None else ([v] if isinstance(v, int) else list(v))
			for v in first]
		tr._align_matched = matched
		tr._align_src_line_of = list(range(n))
		# `distinct` is what the tail boundary actually reads -- NOT _align_matched, which stops at the
		# last cursor advance and so under-reports whatever observe_tail matched afterwards.
		tr._align_stats['distinct'] = set(range(matched + 1) if distinct is None else distinct)
		out, stats = tr.annotate_source(lines, kept=kept)
		bar, per = 0, {}
		for l in out:
			if l.startswith('@measure'):
				bar = int(l.split()[1])
			else:
				per[bar] = per.get(bar, 0) + 1
		return out, stats, per

	cases = (
		('plain', [0, 5, 11, 20], 31, None, 0),
		('backward match clamped', [0, 5, 3, 20], 31, None, 1),
		('bar matched nothing', [0, None, 11, 20], 31, None, 1),
		('trailing gap', [0, 5, None, None], 31, None, 2),
		('first bar missed', [None, 5, 11], 31, None, 0),
		('trimmed to 2 bars', [0, 5, 11, 20], 31, 2, 0),
	)
	for label, first, matched, kept, want_empty in cases:
		out, stats, per = annotate(first, matched, kept)
		nums = [int(l.split()[1]) for l in out if l.startswith('@measure')]
		bars = stats['bars']
		# CONSERVATION, restated for the closing bar line: every source note either lands in exactly one
		# annotated bar or is dropped past that line, and none is counted twice. The old form -- all n
		# notes inside bars -- described the tail bucket, which no longer exists.
		if stats['covered_notes'] + stats['dropped_tail_notes'] != n:
			print(f'  FAIL {label}: covered {stats["covered_notes"]} + dropped '
				f'{stats["dropped_tail_notes"]} != {n}'); ok = False
		if sum(per.values()) != stats['covered_notes']:
			print(f'  FAIL {label}: {sum(per.values())} note_on in the file vs '
				f'{stats["covered_notes"]} covered'); ok = False
		if per.get(0):
			print(f'  FAIL {label}: {per[0]} note_on before the first @measure'); ok = False
		if nums != list(range(1, len(nums) + 1)):
			print(f'  FAIL {label}: measure numbers not contiguous from 1: {nums}'); ok = False
		# The last directive CLOSES the file: one past the annotated bars, and nothing after it.
		if nums != list(range(1, bars + 2)):
			print(f'  FAIL {label}: {len(nums)} directives for {bars} bars + 1 terminator'); ok = False
		if per.get(bars + 1):
			print(f'  FAIL {label}: {per[bars + 1]} note_on after the closing bar line'); ok = False
		if kept and bars != kept:
			print(f'  FAIL {label}: {bars} bars annotated, trim kept {kept}'); ok = False
		if stats['empty_bars'] != want_empty:
			print(f'  FAIL {label}: {stats["empty_bars"]} empty bars, expected {want_empty}'); ok = False

	# The closing line lands one past the FURTHEST match, not past the last bar's opening: a last bar
	# whose boundary is 20 while the run matched out to 30 owns 21..30, and reading max(bounds) + 1
	# instead would leave it holding one note and drop the other 10.
	_out, st, per = annotate([0, 5, 11, 20], 31, distinct=range(31))
	if st['dropped_tail_notes'] != n - 31 or per.get(4, 0) != 11:
		print(f'  FAIL closing line: last bar {per.get(4)} notes, dropped {st["dropped_tail_notes"]} '
			f'(want 11 and {n - 31})'); ok = False
	# ...and a trim overrides it: the dropped bars' source goes with the tail, not to the last kept bar.
	_out, st, per = annotate([0, 5, 11, 20], 31, kept=2, distinct=range(31))
	if per.get(2, 0) != 6 or st['dropped_tail_notes'] != n - 11:
		print(f'  FAIL trimmed close: bar 2 {per.get(2)} notes, dropped {st["dropped_tail_notes"]} '
			f'(want 6 and {n - 11})'); ok = False

	# The no-evidence bar must be the EMPTY one, not its predecessor -- that is what the forward fill
	# buys, and reusing the previous boundary instead would pass every conservation check above while
	# blaming the wrong bar. MEASURED identical on all 20 yt-piano files, so only this case separates them.
	_out, _st, per = annotate([0, None, 11, 20], 31)
	if per.get(2, 0) != 0 or per.get(1, 0) == 0:
		print(f'  FAIL forward fill: bar 2 matched nothing but bars read {per}'); ok = False

	# Boundaries are monotone BY CONSTRUCTION: a bar opens at its smallest match that is not behind the
	# previous bar's boundary. The case that matters is a bar whose FIRST match in generation order jumps
	# to the far end of its own span -- 71XwSVoXOxI bar 21 matched src 232 across a 201..232 bar. Reading
	# that first value and clamping afterwards emptied the next three bars; here the later bars take
	# their own in-range minimum and stay non-empty.
	_out, st, per = annotate([[0, 1], [201, 232, 210], [209, 216], [217, 224], [226, 233]], 240,
		distinct=range(240))
	if st['empty_bars']:
		print(f'  FAIL monotone construction: {st["empty_bars"]} empty bars where a clamp would give 3 '
			f'({per})'); ok = False
	# ...and a bar whose every match IS behind the previous boundary can only empty ITSELF, never a run.
	_out, st, per = annotate([[0, 1], [100, 110], [20, 30], [120, 130]], 240, distinct=range(240))
	if st['empty_bars'] != 1 or per.get(3, 0) != 0:
		print(f'  FAIL backward bar: want exactly bar 3 empty, got {st["empty_bars"]} empty in {per}')
		ok = False

	# A run with no alignment at all must leave the source untouched rather than emit a bar 1 it cannot
	# justify -- annotate_source is reachable with --align-advance off only by a caller bug, but a
	# silently half-annotated source would be worse than an unchanged one.
	tr = SlidingTranslator(None, tk)
	out, stats, _ = SlidingTranslator(None, tk), None, None
	blank, stats = tr.annotate_source(lines)
	if blank != lines or stats['bars']:
		print(f'  FAIL no alignment: {stats["bars"]} bars written onto an unaligned source'); ok = False

	print(f'{"ok  " if ok else "FAIL"} annotate_source: notes conserved, numbering contiguous, '
		f'empty bar lands on the bar that matched nothing ({len(cases)} cases)')
	return ok


def check_annotated_prelude ():
	"""The region before `@measure 1` keeps its declarations and its last sustain, and loses its elapse.

	The prelude is the only part of an annotated source with no bar over it, so its elapse runs state an
	absolute offset the bar coordinate is meant to replace. Two things could go wrong quietly: dropping
	the sustain that bar 1 is played into (the music then reads as pedal-up when it was not), or reaching
	past `@measure 1` and eating elapse from bar 1 itself (which would shift every onset in the file).
	Both produce a well-formed file, so they are asserted rather than eyeballed.
	"""
	ok = True

	def run (label, lines, want):
		nonlocal ok
		got = trim_annotated_prelude(lines)
		if got != want:
			print(f'  FAIL {label}:\n    got  {got}\n    want {want}')
			ok = False
		return got

	body = ['@measure 1', 'E10', 'note_on C1 #48 $50', '@measure 2', 'E20', 'note_on C1 #4a $50']

	# with sustain: the LAST one survives, the elapse before it goes, the elapse AFTER it stays -- that
	# gap is between the pedal and the downbeat, which is the one prelude duration bar 1 is read against.
	run('sustain kept, earlier sustain and elapse dropped',
		['ticks_per_beat 1 e 0', 'format_type 1', 'E070 E6', 'control_change #40 $7f', 'E1',
			'control_change #43 $7f', 'E2', 'control_change #40', 'E1'] + body,
		['ticks_per_beat 1 e 0', 'format_type 1', 'control_change #43 $7f', 'control_change #40',
			'E1'] + body)

	# no sustain: nothing to anchor, so every elapse line in the region goes and the declarations stay.
	run('no sustain, all prelude elapse dropped',
		['ticks_per_beat 1 e 0', 'format_type 1', 'set_tempo 7 a 1 2 0', 'E0e0 E9',
			'control_change #43 $7f', 'E3'] + body,
		['ticks_per_beat 1 e 0', 'format_type 1', 'set_tempo 7 a 1 2 0',
			'control_change #43 $7f'] + body)

	# a sustain AFTER @measure 1 is body, not prelude: it must not become the anchor, and the file's
	# own bar-1 elapse must survive untouched.
	run('sustain after the downbeat is untouched',
		['ticks_per_beat 1 e 0', 'E5', '@measure 1', 'E10', 'control_change #40 $7f', 'E11',
			'note_on C1 #48 $50'],
		['ticks_per_beat 1 e 0', '@measure 1', 'E10', 'control_change #40 $7f', 'E11',
			'note_on C1 #48 $50'])

	# no @measure at all (the unaligned early return): nothing is a prelude, so nothing is dropped.
	unaligned = ['ticks_per_beat 1 e 0', 'E5', 'control_change #40 $7f', 'E6', 'note_on C1 #48 $50']
	run('no @measure 1, source untouched', unaligned, list(unaligned))

	# @measure 1 first line: an empty prelude is not an error.
	run('empty prelude', list(body), list(body))

	# ...and the same three properties on the real annotated files, where the prelude shapes above came
	# from: the body after @measure 1 is byte-identical, and only elapse/sustain lines leave the prelude.
	src_dir = os.path.join(REPO_ROOT, 'tests', 'output', 'yt-annotate-sep1033', 'src')
	files = sorted(glob.glob(os.path.join(src_dir, '*.midiseq2.txt'))) if os.path.isdir(src_dir) else []
	for p in files:
		lines = open(p, encoding='utf-8').read().splitlines()
		if '@measure 1' not in lines:
			continue
		got = trim_annotated_prelude(lines)
		end, gend = lines.index('@measure 1'), got.index('@measure 1')
		name = os.path.basename(p)
		if lines[end:] != got[gend:]:
			print(f'  FAIL {name}: the body after @measure 1 changed'); ok = False
		def kept (ls):
			return [l for l in ls if not is_elapse_line(l) and not is_sustain_line(l)]
		if kept(lines[:end]) != kept(got[:gend]):
			print(f'  FAIL {name}: a prelude declaration was dropped'); ok = False
		if any(is_elapse_line(l) for l in got[:gend][:max(0, gend - 1)]):
			# only the elapse immediately after a kept sustain may remain
			sus = [i for i, l in enumerate(got[:gend]) if is_sustain_line(l)]
			bad = [i for i, l in enumerate(got[:gend]) if is_elapse_line(l) and (not sus or i < sus[-1])]
			if bad:
				print(f'  FAIL {name}: elapse survived at {bad} before the last sustain'); ok = False

	print(f'{"ok  " if ok else "FAIL"} annotated prelude: last sustain kept, elapse dropped, '
		f'bar 1 onward untouched (5 cases + {len(files)} real files)')
	return ok


def check_eom_numbering_rules ():
	"""Bar 1 is the bar the music starts in, and an empty last output bar is empty on the source too.

	Both rules exist because the bar NUMBER is the only thing tying the two arms together, and both
	failure modes conserve every note and so slip past the conservation checks above.

	  leading <eom>   the model can restate the header and close a measure before playing anything.
	                  MEASURED on 00185501a4: one such <eom> put the first note in bar 2 and shifted all
	                  24 bars up by one against a source arm that started at 1.
	  empty last bar  `_align_kept_measures` is `1 + count(<eom>)`, so an output ending on <eom> opens a
	                  final bar holding no note. MEASURED on 0006fcca63: the source arm charged 2270
	                  note_on (81% of the file) to that bar while its output counterpart was 0 lines.
	"""
	tk = Midiseq2Tokenizer()
	ok = True
	eom = tk.eom_id
	note_on = tk.id_by_token.get('note_on')
	tempo = tk.id_by_token.get('set_tempo')

	def tr_with (first, matched, distinct=None, n=40):
		tr = SlidingTranslator(None, tk)
		tr._align_first_src = [[] if v is None else ([v] if isinstance(v, int) else list(v))
			for v in first]
		tr._align_measures = [[1, 0] for _ in first]
		tr._align_matched = matched
		tr._align_src_line_of = list(range(n))
		tr._align_stats['distinct'] = set(range(matched + 1) if distinct is None else distinct)
		return tr

	# --- rule 2: leading <eom> ---
	# two empty measures, then the music. Both must go, and the per-measure evidence must shift with
	# them or the annotation reads bar 3's matches for bar 1.
	tr = tr_with([[], [], [0], [5], [11]], 31)
	tr._align_eom_at = [1, 2, 5]
	out, lead = tr.drop_leading_eom([tempo, eom, eom, note_on, 7, eom, note_on])
	if lead != 2 or out != [tempo, note_on, 7, eom, note_on]:
		print(f'  FAIL leading eom: dropped {lead}, out {out}'); ok = False
	if tr._align_first_src != [[0], [5], [11]]:
		print(f'  FAIL leading eom: evidence not shifted: {tr._align_first_src}'); ok = False
	if tr._align_eom_at != [3]:
		print(f'  FAIL leading eom: eom marks not reindexed: {tr._align_eom_at}'); ok = False

	# an <eom> AFTER the first note is a real bar boundary and must survive untouched.
	tr = tr_with([[0], [5]], 31)
	tr._align_eom_at = [2]
	out, lead = tr.drop_leading_eom([tempo, note_on, eom, note_on])
	if lead != 0 or out != [tempo, note_on, eom, note_on]:
		print(f'  FAIL trailing eom touched: dropped {lead}, out {out}'); ok = False

	# no note at all: nothing to number bars around, so nothing is dropped.
	tr = tr_with([[0]], 31)
	out, lead = tr.drop_leading_eom([tempo, eom, eom])
	if lead != 0 or out != [tempo, eom, eom]:
		print(f'  FAIL noteless output: dropped {lead}, out {out}'); ok = False

	# --- rule 2b: a leading bar whose notes are a RESTATEMENT ---
	# drop_leading_eom cannot see this one: the leading bar HOLDS notes, so the first note fixes bar 1 and
	# that rule returns untouched. Bar 2 matched source 0 too, so the tie in measure_boundaries leaves bar 1
	# owning nothing -- MEASURED on 5 of the 65 published piano0909 pairs.
	tr = tr_with([[0], [0, 4], [11]], 31)
	tr._align_eom_at = [3, 6]
	out, n = tr.drop_restated_leading_bar([tempo, note_on, 7, eom, note_on, 9, eom, note_on])
	if n != 1 or out != [tempo, note_on, 9, eom, note_on]:
		print(f'  FAIL restated leading bar: dropped {n}, out {out}'); ok = False
	if tr._align_first_src != [[0, 4], [11]]:
		print(f'  FAIL restated leading bar: evidence not shifted: {tr._align_first_src}'); ok = False
	if tr._align_eom_at != [3]:
		print(f'  FAIL restated leading bar: eom marks not reindexed: {tr._align_eom_at}'); ok = False

	# bar 2 opening PAST source 0 is a real bar 1 that owns source 0..n: nothing is dropped.
	tr = tr_with([[0], [5], [11]], 31)
	tr._align_eom_at = [3, 6]
	out, n = tr.drop_restated_leading_bar([tempo, note_on, 7, eom, note_on, 9, eom, note_on])
	if n != 0 or out != [tempo, note_on, 7, eom, note_on, 9, eom, note_on]:
		print(f'  FAIL real bar 1 touched: dropped {n}, out {out}'); ok = False

	# bar 2 with NO match is the forward-fill case in measure_boundaries, not a restatement. An empty list
	# must not be read as "matched source 0".
	tr = tr_with([[0], [], [11]], 31)
	tr._align_eom_at = [3, 6]
	out, n = tr.drop_restated_leading_bar([tempo, note_on, 7, eom, note_on, 9, eom, note_on])
	if n != 0:
		print(f'  FAIL no-evidence bar 2 treated as restatement: dropped {n}'); ok = False

	# restated TWICE: each dropped bar re-tests the one it uncovers.
	tr = tr_with([[0], [0], [0, 4], [11]], 31)
	tr._align_eom_at = [3, 6, 9]
	out, n = tr.drop_restated_leading_bar(
		[tempo, note_on, 7, eom, note_on, 9, eom, note_on, 5, eom, note_on])
	if n != 2 or out != [tempo, note_on, 5, eom, note_on]:
		print(f'  FAIL double restatement: dropped {n}, out {out}'); ok = False
	if tr._align_first_src != [[0, 4], [11]]:
		print(f'  FAIL double restatement: evidence not shifted: {tr._align_first_src}'); ok = False

	# one bar, never closed: dropping it would empty the file, so it stays and the coverage gate judges it.
	tr = tr_with([[0], [0]], 31)
	tr._align_eom_at = []
	out, n = tr.drop_restated_leading_bar([tempo, note_on, 7])
	if n != 0 or out != [tempo, note_on, 7]:
		print(f'  FAIL unclosed single bar: dropped {n}, out {out}'); ok = False

	# A forward-filled boundary must not delimit the kept region. When the first DROPPED bar matched
	# nothing, `measure_boundaries` fills its boundary from the next bar that DID match, which can be
	# far ahead; using it would charge the last kept bar every source note in between.
	# MEASURED on 0264fc02da: bars 57-62 matched nothing, bar 57's boundary filled to 383 (bar 62's one
	# match), and the last kept bar -- opening at 330, own furthest 341 -- was given 330..382, 53
	# source note_on over 302 lines against 9-20 in its neighbours.
	#   bars 1-3 kept, own matches 0..2, 5..7, 10..12    bar 4 (dropped) matches NOTHING
	#   bar 5 (dropped) matches 30, so bar 4 fills to 30 -- 17 notes past what bar 3 ever reached
	tr = tr_with([[0, 1, 2], [5, 6, 7], [10, 11, 12], [], [30]], 30, distinct=range(31), n=40)
	ann, st = tr.annotate_source([f'note_on C1 #{40 + i} $40' for i in range(40)], kept=3)
	nums = [int(l.split()[1]) for l in ann if l.startswith('@measure')]
	if nums != [1, 2, 3, 4]:
		print(f'  FAIL filled boundary: directives {nums}'); ok = False
	# bar 3 opens at source 10 and reached 12, so the region ends at 13: 13 source lines, not 30
	if st['covered_notes'] != 13:
		print(f'  FAIL filled boundary: last bar charged {st["covered_notes"]} notes, want 13 '
			f'(its own reach), not the filled boundary at 30'); ok = False
	if st['dropped_tail_notes'] != 40 - 13:
		print(f'  FAIL filled boundary: dropped {st["dropped_tail_notes"]}, want {40 - 13}'); ok = False
	# ...and when the first dropped bar DOES have evidence, its boundary is used unchanged
	tr = tr_with([[0, 1, 2], [5, 6, 7], [10, 11, 12], [20], [30]], 30, distinct=range(31), n=40)
	ann2, st2 = tr.annotate_source([f'note_on C1 #{40 + i} $40' for i in range(40)], kept=3)
	if st2['covered_notes'] != 20:
		print(f'  FAIL real boundary: charged {st2["covered_notes"]} notes, want 20 (bar 4 opens '
			f'there)'); ok = False

	# --- both arms close the same bar ---
	# close_final_measure moves whichever side is short of a boundary, and which side depends on how the
	# run ended. `complete` = the last bar is whole (the trim cut at an <eom>, or end_of_track): the
	# boundary is appended. Otherwise the run stopped mid-bar: that bar is dropped and the <eom> that
	# opened it stays on as the terminator.
	tr = tr_with([[0], [5], [11]], 20)
	out, bars = tr.close_final_measure([note_on, eom, note_on, eom, note_on, 9], complete=True)
	if out[-1] != eom or bars != 3:
		print(f'  FAIL complete: ends {out[-1]}, bars {bars} (want eom and 3)'); ok = False
	tr = tr_with([[0], [5], [11]], 20)
	out, bars = tr.close_final_measure([note_on, eom, note_on, eom, note_on, 9], complete=False)
	if out != [note_on, eom, note_on, eom] or bars != 2:
		print(f'  FAIL incomplete: out {out}, bars {bars} (want the partial bar dropped, 2)'); ok = False
	# a complete output that ALREADY ends on <eom> must not get a second one
	tr = tr_with([[0], [5]], 20)
	out, bars = tr.close_final_measure([note_on, eom], complete=True)
	if out != [note_on, eom] or bars != 2:
		print(f'  FAIL already closed: out {out}, bars {bars}'); ok = False
	# one bar, never closed: truncating to "the last boundary" would empty the file, so it is kept
	tr = tr_with([[0]], 20)
	out, bars = tr.close_final_measure([note_on, 9], complete=False)
	if out != [note_on, 9] or bars != 1:
		print(f'  FAIL single open bar: out {out}, bars {bars}'); ok = False

	# an output that ends on end_of_track is ALREADY closed: that is the corpus's own terminator, and a
	# boundary after it would open a bar past the end of the track.
	eot = tk.id_by_token.get('end_of_track')
	tr = tr_with([[0], [5]], 20)
	out, bars = tr.close_final_measure([note_on, eom, note_on, eot], complete=True)
	if out != [note_on, eom, note_on, eot] or bars != 2:
		print(f'  FAIL ends on end_of_track: out {out}, bars {bars} (want it untouched, 2)'); ok = False

	# --- the source arm closes at the same number, with nothing after it ---
	lines = [f'note_on C1 #{40 + i} $40' for i in range(40)]
	tr = tr_with([[0], [5], [11]], 20, distinct=range(21))
	ann, st = tr.annotate_source(lines, kept=3)
	nums = [int(l.split()[1]) for l in ann if l.startswith('@measure')]
	if nums != [1, 2, 3, 4]:
		print(f'  FAIL source close: directives {nums}, want 3 bars + a terminator'); ok = False
	if ann[-1] != '@measure 4':
		print(f'  FAIL source close: file ends {ann[-1]!r}, want the closing directive last'); ok = False
	if st['dropped_tail_notes'] != 19:
		print(f'  FAIL source close: dropped {st["dropped_tail_notes"]}, want 19'); ok = False
	if st['covered_notes'] + st['dropped_tail_notes'] != st['src_notes']:
		print(f'  FAIL source close: {st["covered_notes"]} + {st["dropped_tail_notes"]} '
			f'!= {st["src_notes"]}'); ok = False

	# ...and the same exception on this arm: a source consumed to its last line ends on the corpus
	# terminator, so no terminal directive is written. The bar COUNT is the same either way -- the
	# terminator was never counted -- so only the trailing line differs.
	lines_eot = [f'note_on C1 #{40 + i} $40' for i in range(40)] + ['end_of_track']
	tr = tr_with([[0], [5], [11]], 39, distinct=range(40))
	ann, st = tr.annotate_source(lines_eot, kept=3)
	nums = [int(l.split()[1]) for l in ann if l.startswith('@measure')]
	if nums != [1, 2, 3]:
		print(f'  FAIL source ends on end_of_track: directives {nums}, want 3 bars and no terminator')
		ok = False
	if ann[-1] != 'end_of_track':
		print(f'  FAIL source ends on end_of_track: file ends {ann[-1]!r}'); ok = False
	if st['bars'] != 3 or st['dropped_tail_notes'] != 0:
		print(f'  FAIL source ends on end_of_track: bars {st["bars"]}, '
			f'dropped {st["dropped_tail_notes"]} (want 3, 0)'); ok = False

	print(f'{"ok  " if ok else "FAIL"} eom numbering: leading empty AND restated bars dropped with their '
		f'evidence, a filled boundary does not delimit the kept region, both arms close the same bar, '
		f'on end_of_track with no directive after it (16 cases)')
	return ok


def check_overgeneration_guards ():
	"""The reuse stop and the density trim fire on over-generation and stay off by default.

	Over-generation is invisible to the miss ratio: AlignConfig['ReuseCost'] is 0.0, so re-matching one
	source note N times books N clean matches and the miss rate reads a PERFECT 0. Both guards here key
	on the source indices instead, and every case below has a miss rate of exactly 0 so that a
	regression to the miss axis cannot pass.

	Bars are replayed through the real `record_align_quality`, and the synthetic index space leaves a
	slot for each <eom>: the bar ordinal is `bisect_left` over eom positions, so an <eom> sharing an
	index with the note after it puts that note in the PREVIOUS bar. That off-by-one silently shifted a
	whole file's bar assignment while I was measuring this, which is why it is spelled out.
	"""
	tk = Midiseq2Tokenizer()

	def replay (bars, **kw):
		'''bars: list of per-bar source-index lists (None = a miss).'''
		tr = SlidingTranslator(None, tk, align_advance=True, **kw)
		idx = 0
		for m, srcs in enumerate(bars):
			if m:
				tr._align_eom_at.append(idx)
				idx += 1
			for s in srcs:
				tr.record_align_quality(idx, s)
				idx += 1
		return tr

	ok = True

	# defaults leave both new axes inert, so an existing command line behaves exactly as before
	base = SlidingTranslator(None, tk)
	if base.align_stop_reuse != 0.0 or base.align_trim_density != 0.0:
		print(f'  FAIL defaults must be off, got reuse={base.align_stop_reuse} '
			f'density={base.align_trim_density}'); ok = False

	# a healthy run: every note its own source note. Neither guard may fire.
	healthy = [list(range(m * 8, m * 8 + 8)) for m in range(6)]
	tr = replay(healthy, align_stop_window=8, align_stop_rate=0.6, align_stop_reuse=0.5,
		align_trim_rate=0.3, align_trim_density=2.0)
	if tr._align_stop_at is not None:
		print(f'  FAIL reuse stop fired on a clean run at {tr._align_stop_at}'); ok = False
	_out, dropped, _f = tr.trim_align_tail(list(range(500)))
	if dropped:
		print(f'  FAIL density trim dropped {dropped} bars of a clean run'); ok = False

	# a tail that pours notes onto source the earlier bars already consumed. Zero misses throughout.
	tail = [list(range(0, 8)), list(range(8, 16)), [16] * 6, [16, 17] * 3]
	tr = replay(tail, align_trim_rate=0.3, align_trim_density=2.0, align_trim_min_notes=4)
	_out, dropped, failed = tr.trim_align_tail(list(range(500)))
	if dropped != 2:
		print(f'  FAIL density trim should drop the 2 over-generated bars, dropped {dropped}'); ok = False
	if failed:
		print('  FAIL a file with 2 healthy bars must not be declared failed'); ok = False
	# and the miss axis alone must NOT see it, which is the whole point
	only_miss = replay(tail, align_trim_rate=0.3)
	if only_miss.trim_align_tail(list(range(500)))[1] != 0:
		print('  FAIL the miss trim was expected to be blind here; the case no longer tests anything')
		ok = False

	# an under-sized final bar is transparent to the walk, not a wall. Without this, a 2-note last bar
	# (the norm -- it is where generation was cut) blocks the drop of everything behind it.
	blocked = [list(range(0, 8)), list(range(8, 16)), [16] * 8, [17] * 8, [18, 18]]
	tr = replay(blocked, align_trim_density=2.0, align_trim_min_notes=4)
	_out, dropped, _f = tr.trim_align_tail(list(range(500)))
	if dropped != 3:
		print(f'  FAIL small last bar should be passed over, expected 3 dropped, got {dropped}'); ok = False

	# truncation lands on an <eom> boundary
	kept = len(tr._align_measures) - dropped
	out, _d, _f = tr.trim_align_tail(list(range(500)))
	if len(out) != tr._align_eom_at[kept - 1]:
		print(f'  FAIL truncation at {len(out)}, expected eom boundary '
			f'{tr._align_eom_at[kept - 1]}'); ok = False

	# nothing survives -> failed, so the caller can refuse to write a fragment
	allbad = [[0] * 6 for _ in range(4)]
	tr = replay(allbad, align_trim_density=2.0, align_trim_min_notes=4)
	_out, dropped, failed = tr.trim_align_tail(list(range(500)))
	if not failed:
		print(f'  FAIL an all-degenerate file must report failed (dropped {dropped})'); ok = False

	# A mid-file collapse the backward walk cannot see. The bars after the loop are HEALTHY, so the
	# walk stops at them and never reaches the damage -- measured on 005c15c173, bars 50-54 repeat one
	# bar and 55-62 recover completely. The run of span-less bars is what finds it.
	#   bars 0-1  healthy, own source
	#   bars 2-6  every match at source 16: each ties with the next, so 2-5 get no span (a run of 4)
	#   bars 7-8  healthy again, source 17.., which is what makes the backward walk stop early
	loop = ([list(range(0, 8)), list(range(8, 16))] + [[16] * 8 for _ in range(5)]
		+ [list(range(17, 25)), list(range(25, 33))])
	tr = replay(loop, align_trim_rate=0.3, align_trim_density=2.0, align_trim_min_notes=4)
	_out, dropped, _f = tr.trim_align_tail(list(range(500)))
	if dropped != 0:
		print(f'  FAIL the tail axes were expected blind to a mid-file loop; the case no longer tests '
			f'anything (dropped {dropped})'); ok = False
	tr = replay(loop, align_trim_rate=0.3, align_trim_density=2.0, align_trim_min_notes=4,
		align_trim_spanless_run=4)
	start = tr.first_spanless_run()
	if start != 2:
		print(f'  FAIL the span-less run should start at bar ordinal 2, got {start}'); ok = False
	_out, dropped, failed = tr.trim_align_tail(list(range(500)))
	if dropped != len(loop) - 2:
		print(f'  FAIL should drop everything from the run on ({len(loop) - 2}), dropped {dropped}')
		ok = False
	if failed:
		print('  FAIL two healthy leading bars must not read as a failed file'); ok = False
	# a lone span-less bar is NORMAL -- the tie convention has to leave one of two bars sharing a
	# boundary empty -- so a run of 1 must not trip a threshold of 4.
	single = [list(range(0, 8)), [8] * 8, [8, 9] * 4, list(range(10, 18)), list(range(18, 26))]
	tr = replay(single, align_trim_spanless_run=4)
	if tr.first_spanless_run() is not None:
		print(f'  FAIL a short run tripped the threshold at {tr.first_spanless_run()}'); ok = False
	# ...and the axis is off by default, so an existing command line is unchanged
	if SlidingTranslator(None, tk).align_trim_spanless_run != 0:
		print('  FAIL align_trim_spanless_run must default to off'); ok = False

	# The span-ratio axis, on the bar the walk would STOP at. The failure it exists for: the last bar
	# replays a 2-note span many times, so the cursor's own tallies read a perfect miss ratio and the
	# density high-water (max(ahead), where the boundary rule uses min(ahead)) also passes it --
	# measured on a4956a41f6, whose last kept bar was seen=9 missed=0 ratio=0.00 with no flag at all,
	# holding 2 source notes against 9 generated.
	#   bars 0-1  healthy, own source
	#   bar 2     8 notes over the 2-note span 16..17, so 4.0x -- and 0 misses, so the miss axis is blind
	narrow = [list(range(0, 8)), list(range(8, 16)), [16, 17] * 4]
	tr = replay(narrow, align_trim_rate=0.3)
	if tr.trim_align_tail(list(range(500)))[1] != 0:
		print('  FAIL the miss axis was expected blind to a narrow-span bar; the case tests nothing')
		ok = False
	tr = replay(narrow, align_trim_rate=0.3, align_trim_span_ratio=4.0)
	ratios = tr.bar_span_ratio()
	_out, dropped, _f = tr.trim_align_tail(list(range(500)))
	if dropped != 1:
		print(f'  FAIL the span-ratio axis should drop the narrow last bar, dropped {dropped} '
			f'(ratios {ratios})'); ok = False
	# the ratio is read off measure_boundaries, so REUSE cannot inflate it: replaying the same span
	# more times must raise the ratio, where the miss tally stays at a perfect 0.
	wider = [list(range(0, 8)), list(range(8, 16)), [16, 17] * 8]
	tr = replay(wider, align_trim_span_ratio=4.0)
	r = tr.bar_span_ratio()[2]
	if r is None or r <= ratios[2]:
		print(f'  FAIL more reuse over the same span must read a higher ratio, got {r} vs {ratios[2]}')
		ok = False
	# a healthy run must not be touched by it, at the value the script ships
	tr = replay(healthy, align_trim_rate=0.3, align_trim_span_ratio=4.0)
	if tr.trim_align_tail(list(range(500)))[1]:
		print('  FAIL span-ratio axis fired on a clean run'); ok = False
	# it is consulted ONLY at the stopping bar, never to cut mid-file: a narrow bar with healthy bars
	# after it must survive, because a false positive there would discard everything behind it.
	shielded = ([list(range(0, 8)), [16, 17] * 4] + [list(range(18, 26)), list(range(26, 34))])
	tr = replay(shielded, align_trim_rate=0.3, align_trim_span_ratio=4.0)
	if tr.trim_align_tail(list(range(500)))[1] != 0:
		print('  FAIL span-ratio must not cut mid-file, only refuse to stop'); ok = False
	# ...and off by default
	if SlidingTranslator(None, tk).align_trim_span_ratio != 0.0:
		print('  FAIL align_trim_span_ratio must default to off'); ok = False

	# the stop needs a SUSTAINED collapse: one bad window must not fire it. This stop is online, and a
	# transient dip is indistinguishable from a real collapse at the moment it fires -- measured, a
	# recoverable blip's bad run was LONGER than a true collapse's, so the hysteresis is the only
	# defence and a regression that drops it would discard healthy bars.
	W = 8
	blip = [list(range(0, 8)), [8] * 8, list(range(9, 17)), list(range(17, 25))]
	tr = replay(blip, align_stop_window=W, align_stop_rate=1.1, align_stop_reuse=0.5)
	if tr._align_stop_at is not None:
		print(f'  FAIL reuse stop fired on a single-window blip at {tr._align_stop_at}'); ok = False
	collapse = [list(range(0, 8))] + [[8] * 8 for _ in range(4)]
	tr = replay(collapse, align_stop_window=W, align_stop_rate=1.1, align_stop_reuse=0.5)
	if tr._align_stop_at is None:
		print('  FAIL reuse stop missed a sustained collapse'); ok = False
	elif tr._align_stop_cause != 'reuse':
		print(f'  FAIL stop cause should be reuse, got {tr._align_stop_cause!r}'); ok = False

	# The reuse axis reads its OWN window, and it has to be longer than the miss axis's. Bar-to-bar
	# repetition is INVISIBLE inside one bar's worth of notes: each bar below replays the same 8 source
	# notes, so an 8-note window is all-distinct (1.00, perfectly healthy) while a 24-note window
	# spanning 3 of them reads 8/24 = 0.33 and crosses 0.34. MEASURED on 005c15c173 bars 50-54, where
	# the model repeated one bar 5 times: every 16-note window read 0.85-1.00 while the run re-used
	# source 1.51x, so the shipped 16-note window could not fire.
	loop = [list(range(0, 8))] + [list(range(8, 16)) for _ in range(7)]
	tr = replay(loop, align_stop_window=W, align_stop_rate=1.1, align_stop_reuse=0.34)
	if tr._align_stop_at is not None:
		print(f'  FAIL a window of {W} was expected blind to bar-to-bar repetition; the case no longer '
			f'tests anything (fired at {tr._align_stop_at})'); ok = False
	tr = replay(loop, align_stop_window=W, align_stop_rate=1.1, align_stop_reuse=0.34,
		align_stop_reuse_window=24)
	if tr._align_stop_at is None or tr._align_stop_cause != 'reuse':
		print(f'  FAIL the widened reuse window missed the repetition, cause '
			f'{tr._align_stop_cause!r}'); ok = False
	# the hysteresis counts MISS windows, not reuse windows: tying it to the reuse window would make
	# widening that window self-cancelling, since a longer window then demands a longer collapse.
	if tr.align_stop_reuse_window != 24 or tr.align_stop_window != W:
		print(f'  FAIL windows not independent: reuse {tr.align_stop_reuse_window}, '
			f'miss {tr.align_stop_window}'); ok = False
	# 0 means "use the miss window", so an existing command line is bit-identical
	same = SlidingTranslator(None, tk, align_advance=True, align_stop_window=W, align_stop_reuse=0.34)
	if same.align_stop_reuse_window != W:
		print(f'  FAIL an unset reuse window must fall back to {W}, got '
			f'{same.align_stop_reuse_window}'); ok = False
	# ...and the widened window must not make the axis trigger-happy: a clean run stays clean
	tr = replay(healthy, align_stop_window=W, align_stop_rate=0.6, align_stop_reuse=0.34,
		align_stop_reuse_window=24)
	if tr._align_stop_at is not None:
		print(f'  FAIL widened reuse window fired on a clean run at {tr._align_stop_at}'); ok = False

	# a run of MISSES must not be reported as reuse: the two axes stay independent so the reported
	# cause is the thing that actually happened
	misses = [list(range(0, 8))] + [[None] * 8 for _ in range(4)]
	tr = replay(misses, align_stop_window=W, align_stop_rate=0.6, align_stop_reuse=0.5)
	if tr._align_stop_at is None or tr._align_stop_cause != 'miss':
		print(f'  FAIL a miss collapse should be attributed to miss, got '
			f'{tr._align_stop_cause!r}'); ok = False

	print(f'{"ok  " if ok else "FAIL"} over-generation guards: reuse stop needs a sustained collapse '
		f'and its own longer window to see bar-to-bar repetition, density trim drops the tail past an '
		f'under-sized bar, the span-less run reaches a mid-file loop, the span ratio catches reuse the '
		f'miss axis cannot see, all off by default')
	return ok


def check_render (root, samples):
	'''render_lines must round-trip a real file's content lines through the vocab.

	Encoding a file's lines and rendering them back should reproduce the content lines exactly (the
	directives are dropped by design, and <eom> comes back as @measure with fresh numbering).
	'''
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	bad = 0
	src = os.path.join(root, 'midi-seq2-score')
	for name in sorted(os.listdir(src))[:samples]:
		lines = open(os.path.join(src, name)).read().splitlines()
		content = [l for l in lines if not l.startswith('@')]
		ids = encode_lines(lines, tk, False)
		back = render_lines(ids, tk, kw)
		if back != content:
			bad += 1
			first = next((i for i, (a, b) in enumerate(zip(back, content)) if a != b), None)
			print(f'  FAIL {name[:8]}: {len(back)} vs {len(content)} lines'
				+ (f', first diff line {first}: {back[first]!r} vs {content[first]!r}'
					if first is not None else ''))
	print(f'{"ok  " if not bad else "FAIL"} render_lines round-trips real content lines '
		f'({samples} files, {bad} mismatches)')
	return bad == 0


def check_no_unknown (root, samples):
	'''No off-vocab tokens in the corpus, so <unknown> in output is always a bug.'''
	tk = Midiseq2Tokenizer()
	total = bad = 0
	for arm in ('midi-seq2-irregular', 'midi-seq2-score'):
		path = os.path.join(root, arm)
		for name in sorted(os.listdir(path))[:samples]:
			ids = encode_lines(open(os.path.join(path, name)).read().splitlines(), tk, True)
			total += len(ids)
			bad += sum(1 for i in ids if i == tk.unknown_id)
	print(f'{"ok  " if not bad else "FAIL"} no <unknown> in corpus '
		f'({total} tokens over {samples} files/arm, {bad} off-vocab)')
	return bad == 0


def check_keywords ():
	'''The keyword set must be exactly the 16 event keywords, derived from the vocab.'''
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	expect = {'ticks_per_beat', 'format_type', 'set_tempo', 'time_signature', 'key_signature',
		'sequence_number', 'channel_prefix', 'smpte_offset', 'end_of_track', 'note_on', 'note_off',
		'polytouch', 'control_change', 'program_change', 'aftertouch', 'pitchwheel'}
	ok = kw == expect
	if not ok:
		print(f'  FAIL missing {expect - kw}, extra {kw - expect}')
	# elapse tokens must not be classified as keywords, or lines would split wrong
	if any(is_elapse(t) for t in kw):
		print('  FAIL an elapse token leaked into the keyword set'); ok = False
	print(f'{"ok  " if ok else "FAIL"} keyword_tokens derives the 16 event keywords from the vocab')
	return ok


def check_header (root):
	'''source_header picks up the leading header lines and stops at content.'''
	name = sorted(os.listdir(os.path.join(root, 'midi-seq2-score')))[0]
	lines = open(os.path.join(root, 'midi-seq2-score', name)).read().splitlines()
	head = source_header(lines)
	ok = len(head) >= 1 and all(h.split()[0] in ('ticks_per_beat', 'format_type') for h in head)
	if not ok:
		print(f'  FAIL header: {head}')
	print(f'{"ok  " if ok else "FAIL"} source_header: {head}')
	return ok


def parse_note_ons (lines):
	'''note_on onsets/pitches read from TEXT, independently of the tokenizer.

	Deliberately a second implementation rather than a helper shared with the tool: it walks the source
	strings while note_on_events walks token ids, so agreement between them is evidence about the token
	walk. Sharing code would make the check pass by construction.
	'''
	notes = []
	abst = 0
	for line in lines:
		parts = line.split()
		if not parts:
			continue
		for p in parts:
			if is_elapse(p):
				abst += int(p[1:], 16)
		if parts[0] != 'note_on':
			continue
		pitch = next((int(p[1:], 16) for p in parts if p.startswith('#')), None)
		if pitch is not None:
			notes.append((abst, pitch))
	return notes


def check_note_on_events (root, samples):
	'''note_on_events must agree with the text walk on every event, in order.

	This is the bridge the whole inspection rests on: an attention row is keyed on a TOKEN index, and a
	musical note is what we want to talk about. If the mapping drifts by one event the figures still
	render, still look plausible, and are wrong.
	'''
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	total, bad, files = 0, 0, 0
	for arm in ('midi-seq2-score', 'midi-seq2-irregular'):
		d = os.path.join(root, 'nota1m-100', arm)
		if not os.path.isdir(d):
			d = os.path.join(root, arm)
		if not os.path.isdir(d):
			continue
		for name in sorted(os.listdir(d))[:samples]:
			lines = open(os.path.join(d, name)).read().splitlines()
			ids = encode_lines(lines, tk)
			events, _, _ = note_on_events(ids, tk, kw)
			expect = parse_note_ons(lines)
			total += len(expect)
			files += 1
			if len(events) != len(expect):
				bad += 1
				print(f'  FAIL {name}: {len(events)} events vs {len(expect)} in text')
				continue
			for e, (onset, pitch) in zip(events, expect):
				if e['onset'] != onset or e['pitch'] != pitch:
					bad += 1
					print(f'  FAIL {name}: event {e["order"]} '
						f'({e["onset"]},{e["pitch"]}) vs text ({onset},{pitch})')
					break
			# the pitch token each event names must really BE that pitch
			for e in events:
				if tk.tokens[ids[e['pitch_index']]] != f'#{e["pitch"]:x}':
					bad += 1
					print(f'  FAIL {name}: pitch_index {e["pitch_index"]} is '
						f'{tk.tokens[ids[e["pitch_index"]]]}, not #{e["pitch"]:x}')
					break
	if not files:
		print('skip note_on_events: no corpus arm found')
		return True
	print(f'{"ok  " if not bad else "FAIL"} note_on_events: {total} events over {files} files '
		f'agree with the text walk ({bad} bad)')
	return not bad


def check_incremental_walk (root, samples):
	'''Walking a stream in CHUNKS must equal walking it whole.

	This is what lets a step plot its own notes the moment it generates them, instead of waiting for the
	finished stream. The failure it guards is specific: a chunk boundary landing between `note_on` and
	its `#XX` would, without carried state, silently drop that note -- no error, just a missing point on
	the figure and a link that resolves to nothing. Chunk sizes 1/2/3 are used deliberately, since they
	are small enough to split every event in the corpus.
	'''
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	d = os.path.join(root, 'nota1m-100', 'midi-seq2-irregular')
	if not os.path.isdir(d):
		d = os.path.join(root, 'midi-seq2-irregular')
	if not os.path.isdir(d):
		print('skip incremental walk: no corpus arm found')
		return True
	ok = True
	for name in sorted(os.listdir(d))[:samples]:
		lines = open(os.path.join(d, name)).read().splitlines()[:300]
		ids = encode_lines(lines, tk)
		whole, tick_w, _ = note_on_events(ids, tk, kw)
		for chunk in (1, 2, 3, 17, 256):
			events, tick, state, walked = [], 0, None, 0
			while walked < len(ids):
				part = ids[walked:walked + chunk]
				fresh, tick, state = note_on_events(part, tk, kw, tick0=tick, state=state,
					index0=walked, order0=len(events))
				events.extend(fresh)
				walked += len(part)
			if events != whole or tick != tick_w:
				ok = False
				where = next((i for i, (a, b) in enumerate(zip(events, whole)) if a != b), None)
				print(f'  FAIL {name[:8]} chunk={chunk}: {len(events)} vs {len(whole)} events, '
					f'tick {tick} vs {tick_w}, first diff at {where}')
				break
	print(f'{"ok  " if ok else "FAIL"} incremental walk: chunked == whole at sizes 1/2/3/17/256 '
		f'(boundaries split events)')
	return ok


def check_line_offsets (root, samples):
	'''line_token_offsets must be exact prefix sums, so a window-local index resolves to a global one.

	Two properties, both needed by the inspector: the offsets are additive (encode_lines has no
	cross-line state), and offsets[c] is where a window starting at line c starts in the global stream.
	'''
	tk = Midiseq2Tokenizer()
	d = os.path.join(root, 'nota1m-100', 'midi-seq2-irregular')
	if not os.path.isdir(d):
		d = os.path.join(root, 'midi-seq2-irregular')
	if not os.path.isdir(d):
		print('skip line_token_offsets: no corpus arm found')
		return True
	ok = True
	for name in sorted(os.listdir(d))[:samples]:
		lines = open(os.path.join(d, name)).read().splitlines()[:400]
		for eom in (False, True):
			offsets = line_token_offsets(lines, tk, eom)
			whole = encode_lines(lines, tk, eom)
			if offsets[-1] != len(whole):
				print(f'  FAIL {name} eom={eom}: offsets end {offsets[-1]} vs {len(whole)} tokens')
				ok = False
				continue
			for c in (0, len(lines) // 3, len(lines) // 2):
				tail = encode_lines(lines[c:], tk, eom)
				if whole[offsets[c]:] != tail:
					print(f'  FAIL {name} eom={eom}: line {c} does not start at offset {offsets[c]}')
					ok = False
					break
	print(f'{"ok  " if ok else "FAIL"} line_token_offsets: prefix sums additive and window-aligned')
	return ok


def check_inspector_indexing (root):
	'''The inspector's index arithmetic, with a stub model instead of a real one.

	What can go wrong here is arithmetic, not learning: the <bos> offset on the first window, the
	'producer' shift by one, and the out_base that turns a window-local generated index into a global
	one. A stub returning uniform attention lets those be asserted exactly — every recorded link must
	name a source pitch token that is really a pitch token, and a generated index that really holds the
	pitch it claims.
	'''
	import torch

	d = os.path.join(root, 'nota1m-100', 'midi-seq2-irregular')
	if not os.path.isdir(d):
		d = os.path.join(root, 'midi-seq2-irregular')
	if not os.path.isdir(d):
		print('skip inspector indexing: no corpus arm found')
		return True
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	name = sorted(os.listdir(d))[0]
	lines = open(os.path.join(d, name)).read().splitlines()[:60]

	class StubBackbone:
		'''Uniform attention over 2 layers x 2 heads, so every key clears any threshold below 1.'''

		def __call__ (self, input_ids=None, attention_mask=None, position_ids=None,
			output_attentions=False):
			T = input_ids.shape[1]
			a = torch.full((1, 2, T, T), 0.5)
			return type('O', (), dict(attentions=(a, a)))()

	class StubModel:
		def __init__ (self):
			self.backbone = StubBackbone()

	ok = True
	src_ids = encode_lines(lines, tk)
	# a plausible generated tail: reuse real tokens so it contains real note_on events
	new_ids = src_ids[:120]
	for head in (True, False):
		for query in ('producer', 'self'):
			insp = AttentionInspector(StubModel(), tk, kw, lines, False, 'cpu',
				query=query, threshold=0.1, top_k=0)
			prefix = ([tk.bos_id] if head else []) + src_ids + [tk.sep_id] + \
				([tk.bos_id] if head else [])
			positions = list(range(len(prefix)))
			insp.observe(0, prefix, positions, new_ids, src_ids, 0, len(lines), head, 0)
			if not insp.links:
				print(f'  FAIL head={head} query={query}: no links recorded')
				ok = False
				continue
			# every source index must be a pitch token in the GLOBAL source stream
			for _, out_index, src_index, _, _ in insp.links:
				if not tk.tokens[insp.src_ids_all[src_index]].startswith('#'):
					print(f'  FAIL head={head} query={query}: source index {src_index} is '
						f'{tk.tokens[insp.src_ids_all[src_index]]}, not a pitch')
					ok = False
					break
				if not tk.tokens[new_ids[out_index]].startswith('#'):
					print(f'  FAIL head={head} query={query}: output index {out_index} is '
						f'{tk.tokens[new_ids[out_index]]}, not a pitch')
					ok = False
					break
			# with out_base non-zero the output indices must shift by exactly that
			insp2 = AttentionInspector(StubModel(), tk, kw, lines, False, 'cpu',
				query=query, threshold=0.1, top_k=0)
			insp2.observe(0, prefix, positions, new_ids, src_ids, 0, len(lines), head, 1000)
			shifted = {(o - 1000, s) for _, o, s, _, _ in insp2.links}
			if shifted != {(o, s) for _, o, s, _, _ in insp.links}:
				print(f'  FAIL head={head} query={query}: out_base did not shift indices uniformly')
				ok = False
	print(f'{"ok  " if ok else "FAIL"} inspector indexing: bos offset, query shift and out_base exact')
	return ok


def check_streaming_plots (root, tmp='/tmp/attn_stream_check'):
	'''Figures must appear DURING the run, and must equal ones drawn after it.

	Two claims, each worth a test rather than an argument:

	  1. once step N returns, its figure is already on disk -- counted as the loop runs, not at the end;
	  2. streaming and deferred agree. That is only true because `output` is append-only, so a step's
	     onsets are final the moment its tokens are appended. If that ever stopped holding, the deferred
	     run (which sees the whole stream) would disagree with the streaming one, and this check is what
	     would notice.
	'''
	import shutil
	import torch

	d = os.path.join(root, 'nota1m-100', 'midi-seq2-irregular')
	if not os.path.isdir(d):
		d = os.path.join(root, 'midi-seq2-irregular')
	if not os.path.isdir(d):
		print('skip streaming plots: no corpus arm found')
		return True
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	lines = open(os.path.join(d, sorted(os.listdir(d))[0])).read().splitlines()[:80]

	class StubBackbone:
		def __call__ (self, input_ids=None, attention_mask=None, position_ids=None,
			output_attentions=False):
			T = input_ids.shape[1]
			a = torch.full((1, 2, T, T), 0.5)
			return type('O', (), dict(attentions=(a, a)))()

	class StubModel:
		def __init__ (self):
			self.backbone = StubBackbone()

	shutil.rmtree(tmp, ignore_errors=True)
	os.makedirs(tmp, exist_ok=True)
	src_ids = encode_lines(lines, tk)
	ok = True

	def run (prefix_path):
		'''Drive three steps the way translate does: extend output, THEN observe.'''
		insp = AttentionInspector(StubModel(), tk, kw, lines, False, 'cpu', threshold=0.1, top_k=4,
			plot_prefix=prefix_path)
		output, counts = [], []
		prime_start = 0
		for step in range(3):
			new_ids = src_ids[step * 40:(step + 1) * 40 + 60]
			prime_ids = output[prime_start:]
			prefix = (([tk.bos_id] if step == 0 else []) + src_ids + [tk.sep_id]
				+ ([tk.bos_id] if step == 0 else []) + prime_ids)
			base = len(output)
			output.extend(new_ids)
			# mimic translate's bookkeeping so the cut marks are exercised, not just defaulted away
			step_prime_start, prime_start = prime_start, base + max(1, len(new_ids) // 2)
			# windows must OVERLAP and end before EOF, as real ones do: a stub that ended every window at
			# len(lines) would make the previous-window mark resolve past the end and silently vanish
			insp.observe(step, prefix, list(range(len(prefix))), new_ids, src_ids,
				4 * step, min(len(lines), 4 * step + 30),
				step == 0, base, output=output, prime_start=step_prime_start,
				next_cursor_real=min(len(lines), 4 * (step + 1)), next_prime_start=prime_start)
			counts.append(len(insp.plot_paths))
		return insp, counts

	live, counts = run(os.path.join(tmp, 'live'))
	# the later steps carry a primer and known cuts; if that bookkeeping stopped reaching the window the
	# figures would silently lose their marks, so assert it arrived
	got_prime = [len(live.resolve_step(s)[1]['prime']) for s in sorted(live.windows)]
	if not any(got_prime):
		print(f'  FAIL no step resolved any primer note: {got_prime}')
		ok = False
	steps_sorted = sorted(live.windows)
	cut_counts = [sum(v is not None for v in live.resolve_step(s)[1]['cuts'].values())
		for s in steps_sorted]
	# step 0 has no previous window and no primer, so it legitimately carries fewer marks; later steps
	# must have all four, and a step that lost them silently would draw a figure that looks fine
	if cut_counts[0] < 1 or not all(c == 4 for c in cut_counts[1:]):
		print(f'  FAIL cut positions per step {cut_counts}; want >=1 then 4')
		ok = False
	first = live.resolve_step(steps_sorted[0])[1]['cuts']
	if first['src_prev'] is not None or first['out_prev'] is not None:
		print(f'  FAIL step 0 invented a previous slide: {first}')
		ok = False
	if counts != sorted(counts) or counts[0] < 1 or counts[-1] != len(live.windows):
		print(f'  FAIL figures did not accumulate one per step: {counts}')
		ok = False
	on_disk = len([f for f in os.listdir(tmp) if f.startswith('live')])
	if on_disk != counts[-1]:
		print(f'  FAIL {counts[-1]} figures reported but {on_disk} on disk')
		ok = False

	late, late_counts = run(None)
	if any(late_counts):
		print(f'  FAIL deferred inspector drew during the run: {late_counts}')
		ok = False
	late.plot_prefix = os.path.join(tmp, 'late')
	for step in sorted(late.windows):
		late.plot_step(step)

	if live.links != late.links:
		print(f'  FAIL streaming and deferred recorded different links '
			f'({len(live.links)} vs {len(late.links)})')
		ok = False
	for step in sorted(live.windows):
		if live.resolve_step(step) != late.resolve_step(step):
			print(f'  FAIL step {step} resolves differently streaming vs deferred')
			ok = False
			break
	n_live = len([f for f in os.listdir(tmp) if f.startswith('live')])
	n_late = len([f for f in os.listdir(tmp) if f.startswith('late')])
	if not n_live or n_live != n_late:
		print(f'  FAIL figure counts differ: {n_live} streamed vs {n_late} deferred')
		ok = False
	shutil.rmtree(tmp, ignore_errors=True)
	print(f'{"ok  " if ok else "FAIL"} streaming plots: accumulate per step {counts}, '
		f'equal to deferred')
	return ok


def check_fallback_links (root):
	'''A note whose every score is below threshold must still keep exactly its ARGMAX.

	Otherwise a diffuse attention row and a row that had nothing to look at both produce no line, and the
	figure cannot tell them apart. The stub gives each key a distinct, deliberately tiny score so the
	expected winner is known in advance: the assertion is not merely "a link exists" but "the link is the
	one with the highest score", which is what makes the fallback trustworthy rather than arbitrary.
	'''
	import torch

	d = os.path.join(root, 'nota1m-100', 'midi-seq2-irregular')
	if not os.path.isdir(d):
		d = os.path.join(root, 'midi-seq2-irregular')
	if not os.path.isdir(d):
		print('skip fallback links: no corpus arm found')
		return True
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	lines = open(os.path.join(d, sorted(os.listdir(d))[0])).read().splitlines()[:60]
	src_ids = encode_lines(lines, tk)
	new_ids = src_ids[:120]

	class RampBackbone:
		'''Tiny scores rising with key index, so the LAST key is every row's argmax.

		Values top out around 1e-3, far under any sane threshold, so nothing can pass and the fallback is
		the only path that can produce a link.
		'''

		def __call__ (self, input_ids=None, attention_mask=None, position_ids=None,
			output_attentions=False):
			T = input_ids.shape[1]
			ramp = torch.arange(T, dtype=torch.float32) / max(1, T - 1) * 1e-3
			a = ramp.view(1, 1, 1, T).expand(1, 2, T, T).contiguous()
			return type('O', (), dict(attentions=(a, a)))()

	class StubModel:
		def __init__ (self):
			self.backbone = RampBackbone()

	ok = True
	insp = AttentionInspector(StubModel(), tk, kw, lines, False, 'cpu',
		threshold=0.05, top_k=8, plot_prefix=None)
	prefix = [tk.bos_id] + src_ids + [tk.sep_id]
	output = list(new_ids)
	insp.observe(0, prefix, list(range(len(prefix))), new_ids, src_ids, 0, len(lines), True, 0,
		output=output, prime_start=0, next_cursor_real=4, next_prime_start=8)

	pairs, window = insp.resolve_step(0)
	gen = len(window['out'])
	if not gen:
		print('  FAIL stub produced no generated notes; the check proves nothing')
		return False
	# exactly one link per generated note, and every one below threshold
	per_note = {}
	for p in pairs:
		per_note.setdefault(p['out_order'], []).append(p)
	if len(per_note) != gen:
		print(f'  FAIL {len(per_note)} of {gen} generated notes kept a fallback link')
		ok = False
	extra = {k: len(v) for k, v in per_note.items() if len(v) != 1}
	if extra:
		print(f'  FAIL some notes kept more than one fallback link: {list(extra.items())[:4]}')
		ok = False
	if any(p['score'] >= 0.05 for p in pairs):
		print('  FAIL a link cleared the threshold; the stub was supposed to make that impossible')
		ok = False
	# the kept link must be the ARGMAX: with a rising ramp that is the last source note in the window
	last_src = max(e['order'] for e in window['src'])
	wrong = [p['out_order'] for p in pairs if p['src_order'] != last_src]
	if wrong:
		print(f'  FAIL {len(wrong)} fallback link(s) did not pick the argmax source note '
			f'(want order {last_src})')
		ok = False
	print(f'{"ok  " if ok else "FAIL"} fallback links: all-sub-threshold row keeps exactly its argmax '
		f'({gen} notes, 1 link each, max score {max(p["score"] for p in pairs):.1e})')
	return ok


def check_eos_mask ():
	'''The first token of a step must never be <eos>, and the override must be reported.

	Three claims, and the third is the one that would rot silently. A model that wants <eos> immediately
	is overridden -- fine, that is the request -- but the run has to SAY so, because a mid-piece override
	means the model wanted to stop and was overruled. If `eos_forced` ever stopped being returned, the
	figures and stats would look like a clean run.

	The stub puts <eos> at an overwhelming logit (matching the observed degenerate case, 13.16 vs 7.71),
	so nothing but the mask can produce a non-empty step. From the SECOND token on, <eos> must still end
	the window normally -- masking every position would make the loop run to max_token.
	'''
	import torch

	tk = Midiseq2Tokenizer()
	ok = True

	class EosModel:
		'''Always wants <eos>; `after` controls from which output position it is allowed to win.'''

		def __init__ (self, vocab, after=0):
			self.vocab, self.after, self.calls = vocab, after, 0

		def __call__ (self, ids, mask, pos):
			self.calls += 1
			logits = torch.full((1, ids.shape[1], self.vocab), 7.71)
			logits[0, -1, tk.eos_id] = 13.16
			# a distinct runner-up, so the forced token is identifiable rather than arbitrary
			logits[0, -1, tk.eom_id] = 9.0
			return logits

	vocab = len(tk.tokens)
	tr = SlidingTranslator(EosModel(vocab), tk, max_token=40)
	ids, forced = tr.generate([tk.bos_id, tk.sep_id], [0, 1], 0.0, 0, 1.0)
	if not ids:
		print('  FAIL step still returned empty; the first-token mask did not fire')
		ok = False
	elif ids[0] == tk.eos_id:
		print('  FAIL first token is <eos> despite the mask')
		ok = False
	elif ids[0] != tk.eom_id:
		print(f'  FAIL forced token is {tk.tokens[ids[0]]!r}, not the runner-up <eom>')
		ok = False
	if not forced:
		print('  FAIL the override was not reported; a mid-piece stop would look like a clean step')
		ok = False
	# <eos> must still terminate from the second token on, so exactly ONE token comes out here
	if len(ids) != 1:
		print(f'  FAIL {len(ids)} token(s) generated; <eos> should end the window after the first')
		ok = False

	# and a step the model did NOT want to end must report forced=False, or the count is meaningless
	class ContentModel:
		def __init__ (self, vocab):
			self.vocab, self.n = vocab, 0

		def __call__ (self, ids, mask, pos):
			self.n += 1
			logits = torch.full((1, ids.shape[1], self.vocab), 1.0)
			# three content tokens, then <eos>
			logits[0, -1, tk.eom_id if self.n <= 3 else tk.eos_id] = 20.0
			return logits

	ids2, forced2 = SlidingTranslator(ContentModel(vocab), tk, max_token=40).generate(
		[tk.bos_id], [0], 0.0, 0, 1.0)
	if forced2:
		print('  FAIL a step that wanted content was reported as forced')
		ok = False
	if len(ids2) != 3:
		print(f'  FAIL expected 3 content tokens before <eos>, got {len(ids2)}')
		ok = False
	print(f'{"ok  " if ok else "FAIL"} first-token <eos> mask: forced 1 token and reported it, '
		f'<eos> still ends the window from token 2, unforced step reports False')
	return ok


def check_primer_and_eom (root):
	'''Primer pitch tokens must be queried, and `<eom>` must be collected on the notes' own tick walk.

	Both are claims about INDEXING, which is where this can go wrong quietly. A primer query reads a row of
	the attention matrix by an index derived from the prefix layout; get it wrong by one and the links still
	appear, still look plausible, and name the wrong note. So the stub is built so the right answer is known
	independently: attention is an identity-ish ramp over keys, and the primer's own token positions are
	computed here from the prefix the test itself assembled.

	The <eom> assertion is the same kind: ticks are checked against a walk of the output stream done here,
	not against the inspector's own numbers.
	'''
	import torch

	d = os.path.join(root, 'nota1m-100', 'midi-seq2-irregular')
	if not os.path.isdir(d):
		d = os.path.join(root, 'midi-seq2-irregular')
	if not os.path.isdir(d):
		print('skip primer/eom: no corpus arm found')
		return True
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	# 400 lines, not 80: the primer and the step must each contain real note_on events, or the check would
	# pass by drawing nothing over nothing
	lines = open(os.path.join(d, sorted(os.listdir(d))[0])).read().splitlines()[:400]
	src_ids = encode_lines(lines, tk)

	class FlatBackbone:
		def __call__ (self, input_ids=None, attention_mask=None, position_ids=None,
			output_attentions=False):
			T = input_ids.shape[1]
			a = torch.full((1, 2, T, T), 0.5)
			return type('O', (), dict(attentions=(a, a)))()

	class StubModel:
		def __init__ (self):
			self.backbone = FlatBackbone()

	ok = True
	insp = AttentionInspector(StubModel(), tk, kw, lines, False, 'cpu', threshold=0.1, top_k=4,
		plot_prefix=None)

	# step 0 lays down a stretch of output with a bar boundary planted in it, so step 1 has a primer that
	# contains a known <eom>
	first = list(src_ids[:300]) + [tk.eom_id] + list(src_ids[300:600])
	output = list(first)
	prefix0 = [tk.bos_id] + list(src_ids) + [tk.sep_id]
	insp.observe(0, prefix0, list(range(len(prefix0))), first, src_ids, 0, 40, True, 0,
		output=output, prime_start=0, next_cursor_real=8, next_prime_start=250)

	# step 1: the primer is output[60:len(output)] and sits at the prefix's tail, as translate builds it
	prime_start, base = 250, len(output)
	prime_ids = output[prime_start:base]
	prefix = list(src_ids) + [tk.sep_id] + list(prime_ids)
	new_ids = list(src_ids[600:900])
	output.extend(new_ids)
	# cursor 0: src_ids is the encoding of lines[0:], so the window's local indices only map onto the
	# global source stream at cursor 0. A mismatched cursor would silently drop every key.
	insp.observe(1, prefix, list(range(len(prefix))), new_ids, src_ids, 0, 44, False, base,
		output=output, prime_start=prime_start, next_cursor_real=12, next_prime_start=base + 30)

	pairs, window = insp.resolve_step(1)
	prime_links = [p for p in pairs if p.get('kind') == 'prime']
	gen_links = [p for p in pairs if p.get('kind') == 'gen']
	if not prime_links:
		print('  FAIL step 1 recorded no primer-context links')
		ok = False
	if not gen_links:
		print('  FAIL step 1 recorded no generated links; the check proves nothing')
		ok = False
	# a primer link must land on a primer note, and a generated link on a generated note -- swapping the two
	# would still produce a full-looking figure
	prime_onsets = {e['onset'] for e in window['prime']}
	out_onsets = {e['onset'] for e in window['out']}
	stray = [p for p in prime_links if p['out_onset'] not in prime_onsets]
	if stray:
		print(f'  FAIL {len(stray)} primer link(s) do not land on a primer note')
		ok = False
	stray = [p for p in gen_links if p['out_onset'] not in out_onsets]
	if stray:
		print(f'  FAIL {len(stray)} generated link(s) do not land on a generated note')
		ok = False
	# every primer note must be covered: the fallback guarantees at least one link per queried row, so a
	# missing note means its row was never queried
	linked = {p['out_onset'] for p in prime_links}
	missed = [e['onset'] for e in window['prime'] if e['onset'] not in linked]
	if missed:
		print(f'  FAIL {len(missed)} of {len(window["prime"])} primer notes have no link')
		ok = False

	# <eom>: ticks checked against an independent walk of the same stream
	want = []
	acc, cur = 0, None
	for i, tid in enumerate(output):
		tok = tk.tokens[tid]
		if is_elapse(tok):
			acc += int(tok[1:], 16)
		elif tok == '<eom>':
			want.append((i, acc))
	if not want:
		print('  FAIL the stub output contains no <eom>; the check proves nothing')
		ok = False
	if insp.out_eoms != want:
		print(f'  FAIL <eom> marks {insp.out_eoms} != independent walk {want}')
		ok = False
	# and the step that DRAWS a boundary must carry it: step 1's primer contains the planted one
	drawn = window.get('eoms') or []
	if not drawn:
		print('  FAIL step 1 resolved no <eom> tick despite one inside its primer')
		ok = False
	elif any(t not in [t for _, t in want] for t in drawn):
		print(f'  FAIL step 1 drew an <eom> tick that is not in the stream: {drawn}')
		ok = False
	print(f'{"ok  " if ok else "FAIL"} primer/eom: {len(prime_links)} primer link(s) over '
		f'{len(window["prime"])} primer notes, {len(gen_links)} generated, '
		f'<eom> {insp.out_eoms} matches an independent walk')
	return ok


def check_plot_layout ():
	'''The step figure must draw what it claims: links crossing into the output panel, primer separated
	from generated by SHAPE, and a dashed cut mark per known boundary on each lane.

	Interrogates the figure objects rather than the image, so a failure names the artist that is wrong
	instead of reporting a pixel difference. The one thing pixels are needed for -- whether the panel
	background still hides the links -- is asserted as the patch being invisible, which the minimal
	mechanism test showed to be exactly the difference between 0 and 148 surviving link pixels.
	'''
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.patches import ConnectionPatch
	from matplotlib.lines import Line2D

	# src_prev deliberately AWAY from the source window's start (480 of 0..960, so x=0.5): an anchor that
	# happens to land on x=0 would pass equally under the old primer-at-zero rule and prove nothing.
	src = [dict(onset=0, pitch=60, order=0), dict(onset=480, pitch=72, order=1),
		dict(onset=960, pitch=65, order=2)]
	out = [dict(onset=1000, pitch=61, order=5), dict(onset=1440, pitch=71, order=6)]
	prime = [dict(onset=200, pitch=55, order=3), dict(onset=600, pitch=57, order=4)]
	window = dict(src=src, out=out, prime=prime, lines=(0, 40),
		eoms=[480, 960, 1440],
		cuts=dict(src_prev=480, src_next=720, out_prev=200, out_next=1300))
	# one passing link, one sub-threshold fallback, one primer-context link: all three renderings
	pairs = [dict(step=0, score=1.0, kind='gen', src_onset=0, src_pitch=60, src_order=0,
			out_onset=1000, out_pitch=61, out_order=5),
		dict(step=0, score=0.01, kind='gen', src_onset=480, src_pitch=72, src_order=1,
			out_onset=1440, out_pitch=71, out_order=6),
		dict(step=0, score=0.4, kind='prime', src_onset=0, src_pitch=60, src_order=0,
			out_onset=200, out_pitch=55, out_order=3)]

	captured = {}
	real = plt.subplots

	def spy (*a, **k):
		fig, axes = real(*a, **k)
		captured['fig'], captured['axes'] = fig, axes
		return fig, axes

	path = '/tmp/attn_layout_check.png'
	plt.subplots = spy
	try:
		plot_attention_step(pairs, window, 0, path, 0.05)
	finally:
		plt.subplots = real
	fig, (ax_s, ax_o) = captured['fig'], captured['axes']
	ok = True

	if ax_s.patch.get_visible() or ax_o.patch.get_visible():
		print('  FAIL a panel background is still visible; it would occlude the links')
		ok = False

	markers = {}
	for coll in ax_o.collections:
		lbl = coll.get_label()
		paths = coll.get_paths()
		markers[lbl] = (len(coll.get_offsets()), len(paths))
	prime_lbl = next((k for k in markers if k.startswith('primer')), None)
	gen_lbl = next((k for k in markers if k.startswith('generated')), None)
	if prime_lbl is None or gen_lbl is None:
		print(f'  FAIL output panel lacks separate primer/generated series: {list(markers)}')
		ok = False
	else:
		if markers[prime_lbl][0] != len(prime) or markers[gen_lbl][0] != len(out):
			print(f'  FAIL wrong point counts: primer {markers[prime_lbl][0]} (want {len(prime)}), '
				f'generated {markers[gen_lbl][0]} (want {len(out)})')
			ok = False
		# distinct SHAPE, not merely distinct colour: compare the marker paths' vertex counts
		p_verts = len(ax_o.collections[0].get_paths()[0].vertices)
		g_verts = len(ax_o.collections[1].get_paths()[0].vertices)
		if p_verts == g_verts:
			print(f'  FAIL primer and generated markers have the same shape ({p_verts} vertices)')
			ok = False
		# the primer must be hollow, so it cannot be mistaken for generated output
		fc = ax_o.collections[0].get_facecolors()
		if len(fc) and fc[0][3] != 0:
			print('  FAIL primer markers are filled; they should be hollow')
			ok = False

	# one dashed cut line per known boundary, on the lane it belongs to. <eom> lines are SOLID, so they do
	# not enter this count -- which is also the point: a bar line and a window cut must not read alike.
	for ax, name, want in ((ax_s, 'source', 2), (ax_o, 'output', 2)):
		dashed = [ln for ln in ax.lines if ln.get_linestyle() not in ('-', 'None')]
		if len(dashed) != want:
			print(f'  FAIL {name} panel has {len(dashed)} dashed cut line(s), want {want}')
			ok = False
	# every <eom> in range gets its own line, and only on the output lane -- they are the model's barring,
	# not the source's
	eom = [ln for ln in ax_o.lines
		if ln.get_linestyle() == '-' and ln.get_color() == '#8a7f5a']
	if len(eom) != len(window['eoms']):
		print(f'  FAIL output panel has {len(eom)} <eom> line(s), want {len(window["eoms"])}')
		ok = False
	if any(ln.get_color() == '#8a7f5a' for ln in ax_s.lines):
		print('  FAIL an <eom> line was drawn on the source panel')
		ok = False

	links = [a for a in ax_s.get_children() if isinstance(a, ConnectionPatch)]
	if len(links) != len(pairs):
		print(f'  FAIL {len(links)} link artist(s) for {len(pairs)} pair(s)')
		ok = False
	elif any(l.get_clip_on() for l in links):
		print('  FAIL a link is clipped; it cannot reach the output panel')
		ok = False
	else:
		# the passing link solid, the sub-threshold fallback dotted and fainter -- so a fallback can never
		# be read as evidence of the same standing
		strong, weak, prime_link = links[0], links[1], links[2]
		if weak.get_linestyle() == strong.get_linestyle():
			print(f'  FAIL fallback link has the same line style as a passing one: '
				f'{weak.get_linestyle()}')
			ok = False
		if weak.get_alpha() >= strong.get_alpha() or weak.get_linewidth() >= strong.get_linewidth():
			print(f'  FAIL fallback link is not fainter/thinner: alpha {weak.get_alpha():.3f} vs '
				f'{strong.get_alpha():.3f}, lw {weak.get_linewidth():.2f} vs '
				f'{strong.get_linewidth():.2f}')
			ok = False
		# a primer-context link must be distinguishable from a production link by colour AND dash: they
		# answer different questions and must not be read as one body of evidence
		if prime_link.get_edgecolor() == strong.get_edgecolor():
			print('  FAIL primer link has the same colour as a production link')
			ok = False
		if prime_link.get_linestyle() in (strong.get_linestyle(), weak.get_linestyle()):
			print(f'  FAIL primer link reuses a production line style: {prime_link.get_linestyle()}')
			ok = False
		# EVERY link is at or above the opacity floor: a link that is counted in the title but invisible on
		# the figure is the figure disagreeing with its own caption
		faint = [l.get_alpha() for l in links if l.get_alpha() < 0.1 - 1e-9]
		if faint:
			print(f'  FAIL {len(faint)} link(s) below the 0.1 alpha floor: {faint}')
			ok = False
	# the title must count the kinds apart, or a step of nothing but fallbacks reads as a full one
	title = fig._suptitle.get_text() if fig._suptitle is not None else ''
	for word in ('fallback', 'primer'):
		if word not in title:
			print(f'  FAIL title does not report the {word} count: {title!r}')
			ok = False

	# ANCHOR: the first GENERATED note sits exactly on the source lane's prev cut. That is the alignment the
	# figure claims, and it is the one thing a reader uses to judge whether generation picked up where the
	# previous window left off -- so it is asserted numerically, not left to the eye.
	o_off = ax_o.collections[0].get_offsets()
	g_off = ax_o.collections[1].get_offsets()
	s_off = ax_s.collections[0].get_offsets()
	src_ons = [e['onset'] for e in src]
	want_anchor = ((window['cuts']['src_prev'] - min(src_ons)) / (max(src_ons) - min(src_ons)))
	got_anchor = float(min(g_off[:, 0]))
	if abs(got_anchor - want_anchor) > 1e-9:
		print(f'  FAIL first generated note at x={got_anchor:.6f}, want the source prev cut at '
			f'x={want_anchor:.6f}')
		ok = False
	# and the anchor must be where the source lane's own prev-cut LINE is, not merely at the same number:
	# the two are drawn by different code paths and only have to disagree once to mislead
	# Found by its ANNOTATION, which is how a reader identifies it too -- "the leftmost dashed line" would
	# silently start testing the wrong cut the moment a window put next before prev, and matplotlib reports
	# both dash patterns as plain '--' so the style is not usable here.
	s_prev = [a for a in ax_s.texts if a.get_text().startswith('prev cut')]
	if len(s_prev) != 1:
		print(f'  FAIL source panel has {len(s_prev)} prev-cut annotation(s), want 1')
		ok = False
	else:
		s_prev_x = float(s_prev[0].xy[0])
		if abs(s_prev_x - got_anchor) > 1e-9:
			print(f'  FAIL source prev-cut mark at x={s_prev_x:.6f} but generation starts at '
				f'x={got_anchor:.6f}')
			ok = False
		if f'@{window["cuts"]["src_prev"]}' not in s_prev[0].get_text():
			print(f'  FAIL prev-cut label names the wrong tick: {s_prev[0].get_text()!r}')
			ok = False
	# LEFT PIN: the output lane's leftmost drawn note -- the primer's first -- lands on x=0, where the source
	# window's first note is. Together with the cut pin above this fixes the lane's scale as well as its
	# offset, so both must hold at once or the map is not the one the docstring describes.
	if abs(float(min(o_off[:, 0]))) > 1e-9:
		print(f'  FAIL primer does not start at x=0: {min(o_off[:, 0])}')
		ok = False
	# The two pins imply the scale, so assert the IMPLIED width rather than a source-scale one: generated span
	# over primer-to-first-generated span, times the anchor's x. Derived independently of the plot's own
	# arithmetic, so agreement is evidence rather than a tautology.
	out_ons = [e['onset'] for e in out]
	pri_ons = [e['onset'] for e in prime]
	gain = want_anchor / (min(out_ons) - min(pri_ons))
	want_w = (max(out_ons) - min(out_ons)) * gain
	got_w = float(max(g_off[:, 0]) - min(g_off[:, 0]))
	if abs(got_w - want_w) > 1e-9:
		print(f'  FAIL generated block spans {got_w:.6f} x, want {want_w:.6f} under the two pins')
		ok = False
	# the primer runs LEFT of the anchor -- it is earlier material, and must not sit on top of generation
	if float(max(o_off[:, 0])) >= got_anchor:
		print(f'  FAIL primer reaches x={max(o_off[:, 0]):.3f}, at or past the anchor {got_anchor:.3f}')
		ok = False
	# the label must name which rule is in force, so a reader is never guessing at the geometry
	if 'pinned to source start' not in ax_o.get_xlabel():
		print(f'  FAIL output label does not name the two-pin alignment: {ax_o.get_xlabel()!r}')
		ok = False
	# the source lane still spans 0..1 over its own window
	if abs(float(min(s_off[:, 0]))) > 1e-9 or abs(float(max(s_off[:, 0])) - 1.0) > 1e-9:
		print(f'  FAIL source lane is not 0..1: {min(s_off[:, 0])}..{max(s_off[:, 0])}')
		ok = False
	if ax_s.get_xlim() != ax_o.get_xlim():
		print(f'  FAIL panels have different x limits: {ax_s.get_xlim()} vs {ax_o.get_xlim()}')
		ok = False
	# every drawn note must be INSIDE the limits, or the figure is hiding points its legend still counts
	x_lo, x_hi = ax_s.get_xlim()
	outside = [float(x) for x in list(s_off[:, 0]) + list(o_off[:, 0]) + list(g_off[:, 0])
		if not (x_lo <= x <= x_hi)]
	if outside:
		print(f'  FAIL {len(outside)} note(s) fall outside the x limits: {outside}')
		ok = False

	# each lane labels its own ticks, extrapolated by its own affine map
	fig.canvas.draw()
	s_lbl = [t.get_text() for t in ax_s.get_xticklabels()]
	o_lbl = [t.get_text() for t in ax_o.get_xticklabels()]
	if s_lbl == o_lbl:
		print(f'  FAIL both lanes carry the same tick labels: {s_lbl}')
		ok = False
	if ax_s.xaxis.get_ticks_position() != 'top':
		print('  FAIL source ticks are not on the top edge')
		ok = False
	plt.close(fig)

	# STEP 0 has no prev cut, so there is no correspondence to anchor to. It must fall back to normalising the
	# output lane over its own extent -- not anchor to something arbitrary, and not crash on the missing key.
	w0 = dict(window)
	w0['cuts'] = dict(src_next=720, out_next=1300)
	captured.clear()
	plt.subplots = spy
	try:
		plot_attention_step(pairs, w0, 0, path, 0.05)
	finally:
		plt.subplots = real
	f0, (a0s, a0o) = captured['fig'], captured['axes']
	o0 = a0o.collections[0].get_offsets()
	s0 = a0s.collections[0].get_offsets()
	if abs(float(min(o0[:, 0]))) > 1e-9 or abs(float(min(s0[:, 0]))) > 1e-9:
		print(f'  FAIL unanchored step does not fall back to x=0 on both lanes: output '
			f'{min(o0[:, 0])}, source {min(s0[:, 0])}')
		ok = False
	# one dashed cut per lane now (only `next` exists), and no anchor wording on the label
	for ax, name in ((a0s, 'source'), (a0o, 'output')):
		dashed = [ln for ln in ax.lines if ln.get_linestyle() not in ('-', 'None')]
		if len(dashed) != 1:
			print(f'  FAIL unanchored {name} panel has {len(dashed)} dashed cut line(s), want 1')
			ok = False
	if 'pinned' in a0o.get_xlabel():
		print(f'  FAIL unanchored step claims a pin in its label: {a0o.get_xlabel()!r}')
		ok = False
	plt.close(f0)

	# NO PRIMER but a cut to anchor to: the left pin has nothing to fix, so the cut pin must be kept -- it is
	# the more specific claim -- and the scale borrowed from the source lane. Dropping BOTH here would lose the
	# handover position for no reason.
	wc = dict(window)
	wc['prime'] = []
	captured.clear()
	plt.subplots = spy
	try:
		plot_attention_step([pairs[0]], wc, 1, path, 0.05)
	finally:
		plt.subplots = real
	fc, (acs, aco) = captured['fig'], captured['axes']
	gc = next(c.get_offsets() for c in aco.collections if c.get_label().startswith('generated'))
	if abs(float(min(gc[:, 0])) - want_anchor) > 1e-9:
		print(f'  FAIL primerless step lost the cut pin: generated starts at {min(gc[:, 0]):.6f}, '
			f'want {want_anchor:.6f}')
		ok = False
	# source scale: one x unit is one source-window span
	want_wc = (max(out_ons) - min(out_ons)) / (max(src_ons) - min(src_ons))
	got_wc = float(max(gc[:, 0]) - min(gc[:, 0]))
	if abs(got_wc - want_wc) > 1e-9:
		print(f'  FAIL primerless step spans {got_wc:.6f} x, want {want_wc:.6f} at the source scale')
		ok = False
	if 'source scale' not in aco.get_xlabel():
		print(f'  FAIL primerless step does not name the source-scale fallback: {aco.get_xlabel()!r}')
		ok = False
	plt.close(fc)

	if os.path.exists(path):
		os.remove(path)
	print(f'{"ok  " if ok else "FAIL"} step figure: backgrounds transparent, primer/generated shapes '
		f'distinct, {2 + 2} cut marks, {len(window["eoms"])} <eom> lines, primer links distinct, '
		f'alpha floor 0.1; two pins (primer start at x=0, first generated on the source prev cut), '
		f'source-scale fallback without a primer, own-extent fallback without a cut')
	return ok


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--root', default=DEFAULT_ROOT)
	ap.add_argument('--samples', type=int, default=12)
	ap.add_argument('--verbose', action='store_true')
	args = ap.parse_args()

	if not os.path.isdir(args.root):
		print(f'corpus not found: {args.root}')
		return 1

	print(f'corpus {args.root}, {args.samples} samples\n')
	results = [
		check_keywords(),
		check_positions(),
		check_absolute_prefix(),
		check_encdec_split_inference(),
		check_encdec_inspector_indexing(args.root),
		check_advance(),
		check_encode_parity(args.root, args.samples, args.verbose),
		check_prefix_parity(args.root, args.samples, args.verbose),
		check_render(args.root, args.samples),
		check_no_unknown(args.root, args.samples),
		check_source_advance(args.root),
		check_source_annotation(args.root),
		check_annotated_prelude(),
		check_eom_numbering_rules(),
		check_overgeneration_guards(),
		check_header(args.root),
		check_note_on_events(args.root, args.samples),
		check_incremental_walk(args.root, min(args.samples, 4)),
		check_line_offsets(args.root, min(args.samples, 4)),
		check_inspector_indexing(args.root),
		check_fallback_links(args.root),
		check_eos_mask(),
		check_primer_and_eom(args.root),
		check_plot_layout(),
		check_streaming_plots(args.root),
	]
	failed = results.count(False)
	print(f'\n{len(results) - failed}/{len(results)} checks passed')
	return 1 if failed else 0


if __name__ == '__main__':
	sys.exit(main())
