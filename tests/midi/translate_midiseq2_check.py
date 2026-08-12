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
	count_note_on, source_header, SlidingTranslator, is_elapse)


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
	]
	failed = results.count(False)
	print(f'\n{len(results) - failed}/{len(results)} checks passed')
	return 1 if failed else 0


if __name__ == '__main__':
	sys.exit(main())
