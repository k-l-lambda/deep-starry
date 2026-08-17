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
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.midi.data.seq2seq2 import Seq2Seq2, _get_file
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer

sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'midi'))
from translateMidiseq2 import (encode_lines, keyword_tokens, positions_for, render_lines,
	plot_attention_step,
	count_note_on, source_header, SlidingTranslator, is_elapse, note_on_events,
	line_token_offsets, AttentionInspector)


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


def check_positions ():
	'''pos_style layouts, and that continuation is a plain +1 run.'''
	ok = True
	flat = positions_for('flat', 3, 2)
	if flat != [0, 1, 2, 3, 4, 5]:
		print(f'  FAIL flat: {flat}'); ok = False
	sep = positions_for('sep', 3, 2)
	if sep != [-4, -3, -2, -1, 0, 1]:
		print(f'  FAIL sep: {sep}'); ok = False
	# source ends at -2, <sep> = -1, target starts at 0
	if sep[2] != -2 or sep[3] != -1 or sep[4] != 0:
		print(f'  FAIL sep boundaries: {sep}'); ok = False
	try:
		positions_for('absolute', 3, 2)
		print('  FAIL absolute should raise'); ok = False
	except ValueError:
		pass
	print(f'{"ok  " if ok else "FAIL"} positions_for: flat / sep layouts, absolute refused')
	return ok


def check_advance ():
	'''advance_output: one measure per step via <eom>, half-window fallback, never stalls.'''
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
	# no <eom> in view -> half of the CURRENT TARGET WINDOW (len(output) - prime_start)
	got = tr.advance_output([10, 11, 12, 13, 14, 15], 0)
	if got != 3:
		print(f'  FAIL fallback: {got} != 3 (half of 6)'); ok = False
	# fallback counts the whole view, not the whole stream: view is 4 long from index 4
	got = tr.advance_output([1, 2, 3, 4, 10, 11, 12, 13], 4)
	if got != 6:
		print(f'  FAIL fallback denominator: {got} != 6 (4 + 4//2)'); ok = False
	# a single-token view must still advance
	got = tr.advance_output([10], 0)
	if got != 1:
		print(f'  FAIL single-token advance: {got} != 1'); ok = False
	# an empty view cannot advance, and must not go backwards
	got = tr.advance_output([10, 11], 2)
	if got != 2:
		print(f'  FAIL empty view: {got} != 2'); ok = False
	print(f'{"ok  " if ok else "FAIL"} advance_output: eom step, half-window fallback, no stall')
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

	src = [dict(onset=0, pitch=60, order=0), dict(onset=480, pitch=72, order=1)]
	out = [dict(onset=1000, pitch=61, order=5), dict(onset=1440, pitch=71, order=6)]
	prime = [dict(onset=200, pitch=55, order=3), dict(onset=600, pitch=57, order=4)]
	window = dict(src=src, out=out, prime=prime, lines=(0, 40),
		eoms=[480, 960, 1440],
		cuts=dict(src_prev=0, src_next=300, out_prev=200, out_next=1300))
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

	# The output lane is normalised from its PRIMER, so its first primer note lands at x=0 -- the same place
	# the source lane's first note lands. That alignment is the whole point: the source cursor advances by
	# the notes that roll out of the primer, so those two are the same musical position.
	o_off = ax_o.collections[0].get_offsets()
	s_off = ax_s.collections[0].get_offsets()
	if abs(float(min(o_off[:, 0])) - 0.0) > 1e-9 or abs(float(min(s_off[:, 0])) - 0.0) > 1e-9:
		print(f'  FAIL lanes are not aligned at x=0: primer min {min(o_off[:, 0])}, '
			f'source min {min(s_off[:, 0])}')
		ok = False
	# generated notes must then sit to the RIGHT of the primer, not on top of it
	g_off = ax_o.collections[1].get_offsets()
	if float(min(g_off[:, 0])) <= float(max(o_off[:, 0])):
		print(f'  FAIL generated notes overlap the primer: generated from {min(g_off[:, 0]):.3f}, '
			f'primer to {max(o_off[:, 0]):.3f}')
		ok = False
	if ax_s.get_xlim() != ax_o.get_xlim():
		print(f'  FAIL panels have different x limits: {ax_s.get_xlim()} vs {ax_o.get_xlim()}')
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
	if os.path.exists(path):
		os.remove(path)
	print(f'{"ok  " if ok else "FAIL"} step figure: backgrounds transparent, primer/generated shapes '
		f'distinct, {2 + 2} cut marks, {len(window["eoms"])} <eom> lines, primer links distinct, '
		f'alpha floor 0.1, lanes aligned at x=0')
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
		check_advance(),
		check_encode_parity(args.root, args.samples, args.verbose),
		check_prefix_parity(args.root, args.samples, args.verbose),
		check_render(args.root, args.samples),
		check_no_unknown(args.root, args.samples),
		check_source_advance(args.root),
		check_header(args.root),
		check_note_on_events(args.root, args.samples),
		check_incremental_walk(args.root, min(args.samples, 4)),
		check_line_offsets(args.root, min(args.samples, 4)),
		check_inspector_indexing(args.root),
		check_fallback_links(args.root),
		check_primer_and_eom(args.root),
		check_plot_layout(),
		check_streaming_plots(args.root),
	]
	failed = results.count(False)
	print(f'\n{len(results) - failed}/{len(results)} checks passed')
	return 1 if failed else 0


if __name__ == '__main__':
	sys.exit(main())
