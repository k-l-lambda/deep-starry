
'''Checks for starry.midi.data.seq2seq2.Seq2Seq2 over the three-arm test202608 corpus.

Run: python tests/midi/seq2seq2_feeder_check.py [--root DIR] [--samples N]

The feeder's whole job is that a crop is MARK-ALIGNED: the source window starts and ends on a
directive naming a score position, and the target window covers the same music because it was found
by that directive's key. So the checks here are mostly about boundaries and losslessness, not about
tensor plumbing:

  1. vocab layout   — <sep> is id 5, vocab_size 582, no other special moved
  2. round-trip     — decoding the source half reproduces the cropped source lines exactly
  3. no <unknown>   — the corpus is fully in-vocabulary, so a miss means an encoder bug
  4. boundaries     — ranges land on marks; <bos> iff the crop starts at the piece start; <eos> ends
                      every target exactly once and never appears in the source; <eom> count
  5. alignment      — boundary keys exist in both files; outward-walk distance reported
  6. head/tail rate — the configured p_head/p_tail actually come out
  7. determinism    — random_crop=False repeats exactly; splits are disjoint and stable
  8. collateBatch   — shapes, padding, target_mask covering exactly the post-<sep> region
  6b. tail degrade  — a tail roll falls back to middle unless both arms end on `end_of_track`
  8b. describe()    — same crop as __getitem__; one <eom> per measure it names (the vis relies on it)
  9. length table   — median/p95/p99/max of T per pairing and line cap, so a config can be sized
 10. line_range    — the per-crop cap varies, honours both ends, and stays deterministic
 11. pos_style     — flat/sep/absolute: ids unchanged, <sep> anchoring, sign separation, pad run
 13. transposition — one offset per sample; pitches on ids, key signatures around the circle of
                     fifths on text; both halves shift together and the 'absolute' axis is corrected
'''

import argparse
import os
import random
import statistics
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from starry.midi.data.seq2seq2 import Seq2Seq2, _get_file	# noqa: E402
import starry.midi.data.seq2seq2 as S					# noqa: E402  (the key-signature helpers)


DEFAULT_ROOT = '/home/camus/data/midi/test202608'
PAIRINGS = [
	('midi-seq2-score', 'midi-seq2'),
	('midi-seq2', 'midi-seq2-irregular'),
	('midi-seq2-score', 'midi-seq2-irregular'),
]

failures = []


def check (name, condition, detail=''):
	if condition:
		print(f'  ok   {name}')
	else:
		print(f'  FAIL {name} {detail}')
		failures.append(f'{name} {detail}'.strip())


def feeder (root, source_dir, target_dir, split='*0/1', **args):
	# A FIXED cap by default, so a length assertion has one number to check against; the sampled range
	# gets its own group (check 10).
	options = dict(source_dir=source_dir, target_dir=target_dir, line_range=128)
	options.update(args)
	(dataset,) = Seq2Seq2.load(root, options, splits=split)
	return dataset


def check_vocab (dataset):
	'''1. The special block is positional — every packed artifact and checkpoint depends on it.'''
	print('\n== 1. vocab layout')
	t = dataset.tokenizer
	check('vocab_size == 582', t.vocab_size == 582, f'got {t.vocab_size}')
	check('<sep> == 5', t.sep_id == 5 and t.tokens[5] == '<sep>', f'got {t.tokens[5]!r}')
	expected = [('<pad>', 0), ('<bos>', 1), ('<eos>', 2), ('<unknown>', 3), ('<eom>', 4)]
	check('other specials unmoved', all(t.tokens[i] == n for n, i in expected),
		f'got {t.tokens[:6]}')


def _content (lines):
	'''The tokens a source half is expected to carry: every non-directive line, flattened.'''
	out = []
	for line in lines:
		if line.startswith('@measure') or line.startswith('@tick'):
			continue
		out.extend(line.split())
	return out


def crops (dataset, samples, rng):
	'''Drive the feeder's own crop/align/assemble steps, yielding everything needed to check the
	result against the files. Going through the internals rather than reverse-engineering the crop
	from the ids is what lets the boundary assertions be exact.'''
	for k in range(samples):
		index = dataset.indices[k % len(dataset.indices)]
		name = dataset.names[index]
		source = _get_file(dataset.source, dataset.arm_source, name, dataset.mark_mode)
		target = _get_file(dataset.source, dataset.arm_target, name, dataset.mark_mode)
		a, z = dataset._pick_crop(source, rng)
		align = dataset._align(source, target, a, z)
		if align is None:
			yield dict(name=name, degenerate=True)
			continue
		ids, sep, positions = dataset._assemble(source, target, a, z, align)
		yield dict(name=name, degenerate=False, source=source, target=target, a=a, z=z,
			align=align, ids=ids, sep=sep, positions=positions)


def check_crops (dataset, samples, rng):
	'''2-5. Losslessness, vocabulary coverage, boundary placement, and mark alignment.'''
	print(f'\n== 2-5. round-trip / <unknown> / boundaries / alignment ({samples} crops)')
	t = dataset.tokenizer
	bad = dict(degenerate=0, roundtrip=0, unknown=0, mark=0, bos=0, eos=0, eom=0, inverted=0,
		lines=0, key=0, walk=0, cover=0)
	walks = []
	lengths = []
	for case in crops(dataset, samples, rng):
		if case['degenerate']:
			bad['degenerate'] += 1
			continue
		source, target, a, z = case['source'], case['target'], case['a'], case['z']
		ids, sep = case['ids'], case['sep']
		lengths.append(len(ids))
		s_start, s_end = dataset._bounds(source, a, z)
		t_start, t_end = case['align']

		if t.unknown_id in ids:
			bad['unknown'] += 1
		if t_end <= t_start:
			bad['inverted'] += 1
		if s_end - s_start > dataset.max_lines and z - a > 1:
			# a single mark whose own span exceeds the cap is unavoidable; more than one is a bug.
			bad['lines'] += 1

		# A boundary must sit on a mark line, or on an edge of the file.
		mark_lines = {line for line, _ in source.marks}
		if not (s_start == 0 or s_start in mark_lines):
			bad['mark'] += 1
		if not (s_end == len(source.lines) or s_end in mark_lines):
			bad['mark'] += 1

		# The keys the alignment used must exist in BOTH files (that is what the outward walk buys),
		# and the walk must not move when the key is already shared — a walk that steps past its own
		# mark silently shifts the target window onto neighbouring music.
		if a > 0:
			left = dataset._walk_out(source, target, a, -1)
			if left >= 0:
				walks.append(a - left)
				if source.marks[left][1] not in target.lines_of:
					bad['key'] += 1
				if a - left < 0 or (source.marks[a][1] in target.lines_of and left != a):
					bad['walk'] += 1
		if z < len(source.marks):
			right = dataset._walk_out(source, target, z, 1)
			if right < len(source.marks):
				walks.append(right - z)
				if source.marks[right][1] not in target.lines_of:
					bad['key'] += 1
				if right - z < 0 or (source.marks[z][1] in target.lines_of and right != z):
					bad['walk'] += 1

		# THE central property: the target window must cover the same music as the source window. Every
		# mark key inside the source range that the target has at all must appear inside the target
		# range — otherwise the two halves describe different bars and the pairing is noise. Checked as
		# a superset because the outward walk is allowed to widen the target, never to shift it.
		source_keys = {key for line, key in source.marks if s_start <= line < s_end}
		target_keys = {key for line, key in target.marks if t_start <= line < t_end}
		shared = {key for key in source_keys if key in target.lines_of}
		if not shared <= target_keys:
			bad['cover'] += 1

		tokens = [t.tokens[i] for i in ids]
		src_tokens, tgt_tokens = tokens[:sep], tokens[sep + 1:]
		head, tail = a <= 0, z >= len(source.marks)

		# 2. the source half must decode back to exactly the cropped lines. The source carries no <eos>,
		# so only a leading <bos> has to be stripped before comparing.
		expect = _content(source.lines[s_start:s_end])
		got = src_tokens[1 if head else 0:]
		if got != expect:
			bad['roundtrip'] += 1

		# 4. <bos> is conditional and symmetric: both halves iff the crop reached the piece start.
		if (src_tokens[:1] == ['<bos>']) != head or (tgt_tokens[:1] == ['<bos>']) != head:
			bad['bos'] += 1
		# <eos> is unconditional and target-only — it terminates the generated half, so it must be there
		# for EVERY crop (mid-piece ones included) and must never appear in the source.
		if tgt_tokens[-1:] != ['<eos>'] or '<eos>' in src_tokens:
			bad['eos'] += 1
		# ...and exactly once, at the very end: a stray one inside would be a premature stop signal.
		if tgt_tokens.count('<eos>') != 1:
			bad['eos'] += 1

		# 4. one <eom> per target @measure in range, except @measure 1.
		want_eom = sum(1 for line in target.lines[t_start:t_end]
			if line.startswith('@measure') and line.split()[1] != '1')
		if tgt_tokens.count('<eom>') != want_eom:
			bad['eom'] += 1

	check('no degenerate alignment', bad['degenerate'] == 0, f"{bad['degenerate']} crops")
	check('no <unknown> emitted', bad['unknown'] == 0, f"{bad['unknown']} crops")
	check('target range non-empty', bad['inverted'] == 0, f"{bad['inverted']} crops")
	check('source half round-trips exactly', bad['roundtrip'] == 0, f"{bad['roundtrip']} crops")
	check('boundaries land on marks or file edges', bad['mark'] == 0, f"{bad['mark']} boundaries")
	check('alignment keys exist in both files', bad['key'] == 0, f"{bad['key']} keys")
	check('walk stays put when the key is already shared', bad['walk'] == 0, f"{bad['walk']} walks")
	check('target window covers the source window', bad['cover'] == 0, f"{bad['cover']} crops")
	check('<bos> iff crop starts at the piece start', bad['bos'] == 0, f"{bad['bos']} crops")
	check('<eos> ends every target exactly once, never in the source', bad['eos'] == 0,
		f"{bad['eos']} crops")
	check('<eom> count matches target @measure count', bad['eom'] == 0, f"{bad['eom']} crops")
	check('source stays within the line cap', bad['lines'] == 0, f"{bad['lines']} crops")
	if walks:
		nonzero = [w for w in walks if w]
		print(f'  outward walk: {len(nonzero)}/{len(walks)} boundaries needed one; '
			f'median {statistics.median(nonzero) if nonzero else 0:.0f} max {max(walks)}')
	return lengths


def check_head_tail (dataset, samples, rng):
	'''6. The configured edge probabilities have to actually come out of _pick_crop.'''
	print(f'\n== 6. head/tail frequency ({samples} draws, configured '
		f'{dataset.p_head:.2f}/{dataset.p_tail:.2f})')
	index = dataset.indices[0]
	source = _get_file(dataset.source, dataset.arm_source, dataset.names[index], dataset.mark_mode)
	target = _get_file(dataset.source, dataset.arm_target, dataset.names[index], dataset.mark_mode)
	# Passed through as production does: a tail roll degrades on a file with no terminator, and every
	# file in this corpus has one, so the rate must still land on p_tail. Check 6b covers the other side.
	check('both arms of the sampled file end on end_of_track',
		source.ends_on_terminator and target.ends_on_terminator,
		f'source {source.ends_on_terminator} target {target.ends_on_terminator}')
	head = tail = 0
	for _ in range(samples):
		a, z = dataset._pick_crop(source, rng, target)
		if a <= 0:
			head += 1
		if z >= len(source.marks):
			tail += 1
	p_head, p_tail = head / samples, tail / samples
	print(f'  observed head {p_head:.3f} tail {p_tail:.3f}')
	# A mid-piece draw can land on mark 0 or run to EOF by chance, so the observed rate is a floor
	# plus that slack; 0.05 absolute is comfortably inside it for a corpus this size.
	check('head rate near p_head', abs(p_head - dataset.p_head) < 0.05, f'{p_head:.3f}')
	check('tail rate near p_tail', abs(p_tail - dataset.p_tail) < 0.05, f'{p_tail:.3f}')


def check_tail_degrades (dataset, samples, rng):
	'''6b. A tail roll must degrade to middle when an arm does not end on `end_of_track`.

	The tail mode exists to show the model a real ending. A file closed by `close_final_measure`'s bare
	`@measure N` has none -- that line becomes an ordinary `<eom>` -- so pinning z to EOF there would
	teach that a piece ends at an arbitrary bar line. Asserted by STUBBING the flag rather than by
	finding such a file, so the check does not depend on which corpus is on disk.
	'''
	print(f'\n== 6b. a tail roll degrades without a terminator ({samples} draws)')
	index = dataset.indices[0]
	name = dataset.names[index]
	source = _get_file(dataset.source, dataset.arm_source, name, dataset.mark_mode)
	target = _get_file(dataset.source, dataset.arm_target, name, dataset.mark_mode)
	if not source.marks:
		print('  skipped: the sampled file has no marks in this mode')
		return

	def rate (src_ends, tgt_ends):
		src_was, tgt_was = source.ends_on_terminator, target.ends_on_terminator
		source.ends_on_terminator, target.ends_on_terminator = src_ends, tgt_ends
		try:
			seed = random.Random(20260917)
			hit = sum(1 for _ in range(samples)
				if dataset._pick_crop(source, seed, target)[1] >= len(source.marks))
			return hit / samples
		finally:
			source.ends_on_terminator, target.ends_on_terminator = src_was, tgt_was

	both = rate(True, True)
	neither = rate(False, False)
	src_only = rate(True, False)
	tgt_only = rate(False, True)
	print(f'  tail rate -- both {both:.3f}  neither {neither:.3f}  '
		f'source only {src_only:.3f}  target only {tgt_only:.3f}')
	# Not 0: a middle draw can still grow to EOF by chance. What matters is that it drops to the
	# incidental rate, i.e. well below the p_tail the roll asked for.
	floor = both - dataset.p_tail / 2
	check('tail survives when both arms end on the terminator', both > dataset.p_tail * 0.8, f'{both:.3f}')
	check('tail degrades when neither arm does', neither < floor, f'{neither:.3f} vs both {both:.3f}')
	check('tail degrades when only the source does', src_only < floor, f'{src_only:.3f}')
	check('tail degrades when only the target does', tgt_only < floor, f'{tgt_only:.3f}')
	# The roll is drawn before the test, so degrading must not shift the rng stream.
	a_seed, b_seed = random.Random(4), random.Random(4)
	source.ends_on_terminator = target.ends_on_terminator = False
	try:
		degraded = [dataset._pick_crop(source, a_seed, target) for _ in range(40)]
	finally:
		source.ends_on_terminator = target.ends_on_terminator = True
	kept = [dataset._pick_crop(source, b_seed, target) for _ in range(40)]
	differ = sum(1 for x, y in zip(degraded, kept) if x != y)
	check('degrading changes only the crops that rolled tail', 0 < differ < 40, f'{differ}/40 differ')

	# The flag itself, read off the text.
	blank = _get_file.__globals__['_File']('note_on #3c $40\nend_of_track\n\n', dataset.mark_mode)
	bare = _get_file.__globals__['_File']('note_on #3c $40\n@measure 7\n', dataset.mark_mode)
	check('ends_on_terminator ignores a trailing blank line', blank.ends_on_terminator)
	check('ends_on_terminator is False on a bare @measure close', not bare.ends_on_terminator)


def check_determinism (root, source_dir, target_dir):
	'''7. Val must repeat exactly, and the splits must not overlap.'''
	print('\n== 7. determinism and splits')
	a = feeder(root, source_dir, target_dir, split='19/20', random_crop=False)
	b = feeder(root, source_dir, target_dir, split='19/20', random_crop=False)
	same = all(torch.equal(a[i][0], b[i][0]) and a[i][1] == b[i][1] for i in range(len(a)))
	check('random_crop=False repeats exactly', same)

	varies = len({tuple(a[i][0].tolist()[:12]) for i in range(len(a))}) > 1
	check('deterministic crops still differ per sample', varies)

	train, val = Seq2Seq2.load(root, dict(source_dir=source_dir, target_dir=target_dir),
		splits='*0..18/20:19/20')
	overlap = set(train.indices) & set(val.indices)
	check('train/val disjoint', not overlap, f'{len(overlap)} shared')
	check('splits cover the corpus', len(train.indices) + len(val.indices) == len(train.names),
		f'{len(train.indices)}+{len(val.indices)} != {len(train.names)}')
	check('file list is sorted', train.names == sorted(train.names))


def check_describe (dataset):
	'''8b. describe() must report the SAME crop __getitem__ returns, with usable measure numbers.

	Visualization (tests/midi/seq2seq2_onset_compare.ipynb) labels bar lines from describe()'s measure
	list, pairing the k-th <eom> with the k-th @measure. That pairing is only sound if the counts match
	and the ids are the very ones the model would see, so both are checked here rather than trusted.
	'''
	print('\n== 8b. describe()')
	# A random_crop feeder redraws on every call by design, so the ids/describe() comparison is only
	# meaningful on a deterministic one — rebuild this pairing with random_crop off.
	dataset = feeder(os.path.dirname(dataset.source_root),
		os.path.basename(dataset.source_root), os.path.basename(dataset.target_root),
		mark_mode=dataset.mark_mode, source_eom=dataset.source_eom, random_crop=False)
	t = dataset.tokenizer
	ok_same = ok_count = ok_range = ok_order = True
	for i in range(min(8, len(dataset))):
		index = dataset.indices[i]
		case = dataset.describe(index)
		ids, sep, positions, skip = dataset[i]
		# describe() re-runs the crop; a deterministic feeder must land on the same one.
		if ids.tolist() != case['ids'] or sep != case['sep']:
			ok_same = False
		tokens = [t.tokens[k] for k in case['ids']]
		halves = ((tokens[:case['sep']], case['source_measures'], dataset.source_eom),
			(tokens[case['sep'] + 1:], case['target_measures'], True))
		for half, measures, eom in halves:
			# one <eom> per @measure in range, @measure 1 excepted — the notebook's k-th pairing.
			want = [m for _, m in measures if m != 1]
			if half.count('<eom>') != (len(want) if eom else 0):
				ok_count = False
			if want != sorted(want):
				ok_order = False
		s_start, s_end = case['source_range']
		t_start, t_end = case['target_range']
		if not (0 <= s_start < s_end <= len(case['source'].lines)
				and 0 <= t_start < t_end <= len(case['target'].lines)):
			ok_range = False
		if any(not (s_start <= line < s_end) for line, _ in case['source_measures']) \
				or any(not (t_start <= line < t_end) for line, _ in case['target_measures']):
			ok_range = False
	check('describe() ids match __getitem__', ok_same)
	check('<eom> count matches the reported measures', ok_count)
	check('measure numbers ascend', ok_order)
	check('reported line ranges are sane and contain their measures', ok_range)


def check_collate (dataset):
	'''8. The batch contract: shapes, padding, and the supervised region.'''
	print('\n== 8. collateBatch')
	items = [dataset[i] for i in range(min(4, len(dataset)))]
	batch = dataset.collateBatch(items)
	t = dataset.tokenizer
	width = max(len(ids) for ids, *_ in items)
	check('keys', set(batch) == {'input_ids', 'masks', 'target_mask', 'sep_index', 'position_ids'},
		str(set(batch)))
	check('input_ids shape', tuple(batch['input_ids'].shape) == (len(items), width),
		str(tuple(batch['input_ids'].shape)))
	check('all long dtype', all(v.dtype == torch.long for v in batch.values()))
	ok_pad = ok_mask = ok_target = ok_sep = True
	ok_skip = all(skip == 0 for *_, skip in items)
	for row, (ids, sep, positions, skip) in enumerate(items):
		length = len(ids)
		if length < width and not (batch['input_ids'][row, length:] == t.pad_id).all():
			ok_pad = False
		if batch['masks'][row].sum().item() != length:
			ok_mask = False
		# the supervised region is exactly (sep + skip, length); skip is 0 unless start_jitter drew a
		# nonzero offset for this crop, which this feeder does not enable — see section 12.
		if batch['target_mask'][row].sum().item() != length - sep - 1 - skip:
			ok_target = False
		if batch['target_mask'][row, :sep + 1 + skip].any():
			ok_target = False
		if batch['input_ids'][row, sep].item() != t.sep_id or batch['sep_index'][row].item() != sep:
			ok_sep = False
	check('right-padded with <pad>', ok_pad)
	check('masks count real tokens', ok_mask)
	check('target_mask covers exactly the post-<sep> region', ok_target)
	check('sep_index points at <sep>', ok_sep)
	check('skip is 0 without start_jitter', ok_skip)


def check_line_range (root, source_dir, target_dir, samples, rng):
	'''10. The sampled line cap: it must actually vary, respect both ends, and stay deterministic.

	A greedy crop fills whatever cap it is given, so with a FIXED cap nearly every window comes out
	near-maximal (measured: 0.95 of the cap, p10 0.83). Drawing the cap per crop is what puts short
	windows in the training distribution, so what is checked is the spread, not just the bound.
	'''
	print('\n== 10. sampled line range')
	lo, hi = 20, 256
	dataset = feeder(root, source_dir, target_dir, mark_mode='tick', line_range=[lo, hi])
	check('line_range parsed', dataset.line_range == (lo, hi), str(dataset.line_range))
	check('max_lines is the upper bound', dataset.max_lines == hi, str(dataset.max_lines))

	spans = []
	over = 0
	for _ in range(samples):
		index = dataset.indices[len(spans) % len(dataset.indices)]
		source = _get_file(dataset.source, dataset.arm_source, dataset.names[index], dataset.mark_mode)
		a, z = dataset._pick_crop(source, rng)
		start, end = dataset._bounds(source, a, z)
		spans.append(end - start)
		# One mark whose own span exceeds the cap is unavoidable; a MULTI-mark crop overshooting is not.
		if end - start > hi and z - a > 1:
			over += 1
	spans.sort()
	check('no multi-mark crop exceeds hi', over == 0, f'{over} crops')
	# The whole point: lengths must spread across the range rather than pile up at the top. With a fixed
	# cap the median sits at ~0.95 of it, so a median below 0.75 of hi is the observable difference.
	median = statistics.median(spans)
	check('lengths spread below the cap', median < hi * 0.75, f'median {median:.0f} of hi {hi}')
	check('short crops actually occur', spans[0] <= lo * 2, f'shortest {spans[0]}')
	print(f'  crop lines: min {spans[0]} median {median:.0f} '
		f'p95 {spans[int(len(spans) * .95)]} max {spans[-1]}')

	# A drawn cap must not cost determinism: the draw comes off the same rng as the mode and the start.
	a = feeder(root, source_dir, target_dir, line_range=[lo, hi], split='19/20', random_crop=False)
	b = feeder(root, source_dir, target_dir, line_range=[lo, hi], split='19/20', random_crop=False)
	same = all(torch.equal(a[i][0], b[i][0]) for i in range(len(a)))
	check('a sampled cap stays deterministic under random_crop=False', same)
	varies = len({len(a[i][0]) for i in range(len(a))}) > 1
	check('deterministic crops still differ in length', varies)

	# A scalar must still mean a fixed cap.
	fixed = feeder(root, source_dir, target_dir, line_range=128)
	check('a scalar means a fixed cap', fixed.line_range == (128, 128), str(fixed.line_range))
	for bad in ([0, 10], [200, 100], [1, 2, 3], 0):
		try:
			feeder(root, source_dir, target_dir, line_range=bad)
			check(f'rejects line_range={bad!r}', False, 'accepted')
		except ValueError:
			check(f'rejects line_range={bad!r}', True)


def check_start_jitter (root, source_dir, target_dir):
	'''12. start_jitter: the source crop's start moves, the target's does not, and head crops are exempt.

	The augmentation exists because every crop otherwise begins exactly ON a mark line, which inference
	cannot reproduce — a sliding window over a production file starts mid-measure. So what is checked is
	that the offset is actually applied off-mark, that it leaves the TARGET window alone (that alignment
	is the supervision signal), and that a == 0 crops are untouched, since a == 0 IS the <bos> condition.
	'''
	print('\n== 12. start_jitter augmentation')
	kw = dict(mark_mode='tick', line_range=[20, 256], split='0/1', random_crop=False)
	off = feeder(root, source_dir, target_dir, **kw)
	check('default is off', off.start_jitter == 0.0, str(off.start_jitter))

	explicit = feeder(root, source_dir, target_dir, start_jitter=0.0, **kw)
	same = all(off.describe(i)['ids'] == explicit.describe(i)['ids'] for i in off.indices)
	check('start_jitter=0 is bit-identical to the default', same)

	# with it off, a non-head crop must start exactly on a mark line
	marks_hit = total = 0
	for index in off.indices:
		case = off.describe(index)
		if case['head']:
			continue
		total += 1
		if case['source_range'][0] in {line for line, _ in case['source'].marks}:
			marks_hit += 1
	check('off: every non-head crop starts on a mark', marks_hit == total, f'{marks_hit}/{total}')

	std = 8.0
	on = feeder(root, source_dir, target_dir, start_jitter=std, **kw)
	offsets, head_bad, range_bad, empty = [], 0, 0, 0
	for index in on.indices:
		case = on.describe(index)
		start, end = case['source_range']
		if case['head']:
			# a jitter drawn for an interior crop must not survive onto a head one
			if case['jitter'] != 0 or start != 0:
				head_bad += 1
			continue
		offsets.append(case['jitter'])
		if start < 0 or start >= end:
			range_bad += 1
		if case['sep'] == 0:
			empty += 1
	check('head crops keep jitter 0 and start 0', head_bad == 0, f'{head_bad} bad')
	check('the source range stays valid', range_bad == 0, f'{range_bad} bad')
	check('no crop gets an empty source half', empty == 0, f'{empty} empty')
	check('offsets are actually applied', sum(1 for j in offsets if j) > len(offsets) * 0.5,
		f'{sum(1 for j in offsets if j)}/{len(offsets)} nonzero')
	# a normal draw: mean near 0 and sample std near the requested one. Loose bounds — this asserts the
	# distribution is the right shape, not that a finite sample matches it exactly.
	mean, sigma = statistics.mean(offsets), statistics.pstdev(offsets)
	check('offsets center on the mark', abs(mean) < std * 0.5, f'mean {mean:+.2f}')
	check('offset spread matches the requested std', abs(sigma - std) < std * 0.5,
		f'std {sigma:.2f} vs {std}')
	print(f'  offsets: n {len(offsets)} mean {mean:+.2f} std {sigma:.2f} '
		f'range [{min(offsets)}, {max(offsets)}]')

	# the target window is alignment-derived and must not move with the source's start
	unchanged = sum(1 for i in on.indices
		if on.describe(i)['target_range'] == off.describe(i)['target_range'])
	check('the target range is unaffected', unchanged == len(on.indices),
		f'{unchanged}/{len(on.indices)}')

	twin = feeder(root, source_dir, target_dir, start_jitter=std, **kw)
	stable = all(on.describe(i)['ids'] == twin.describe(i)['ids'] for i in on.indices)
	check('jittered crops stay deterministic under random_crop=False', stable)

	for bad in (-1.0, -0.5):
		try:
			feeder(root, source_dir, target_dir, start_jitter=bad, **kw)
			check(f'rejects start_jitter={bad!r}', False, 'accepted')
		except ValueError:
			check(f'rejects start_jitter={bad!r}', True)

	check_jitter_supervision(on, off)


def check_jitter_supervision (on, off):
	'''12b. The unsupervised head: an offset crop cannot be asked to produce its first bar.

	A crop whose source start moved is missing the head of its first bar, so that bar's target tokens
	are not derivable from the context — supervising them teaches invention. `skip` drops them, up to
	and including the first <eom>. What must hold: skip is keyed on the SAMPLED offset (a crop that drew
	exactly 0 keeps full supervision), it lands just past the first <eom>, it never empties the mask,
	and collateBatch honours it.
	'''
	print('\n== 12b. jitter drops the target\'s first bar from supervision')
	t = on.tokenizer
	ok_key = ok_pos = ok_nonempty = True
	skips, jittered = [], 0
	for index in on.indices:
		case = on.describe(index)
		target_ids, skip, jitter = case['ids'][case['sep'] + 1:], case['skip'], case['jitter']
		# keyed on the sampled offset, not on the start_jitter setting
		if bool(skip) != bool(jitter):
			ok_key = False
		if jitter:
			jittered += 1
			skips.append(skip)
			# exactly one past the first <eom>, so the boundary token itself is unsupervised too
			if skip != target_ids.index(t.eom_id) + 1:
				ok_pos = False
			# describe() cancels the jitter when the half has no <eom>, so a nonzero skip can never
			# consume the whole target half
			if skip >= len(target_ids):
				ok_nonempty = False
	check('skip is nonzero exactly when the sampled offset is', ok_key)
	check('skip lands one past the first <eom>', ok_pos)
	check('skip never consumes the whole target half', ok_nonempty)
	check('an offset crop is present to check', jittered > 0, f'{jittered} jittered')
	if skips:
		print(f'  skip: n {len(skips)} median {statistics.median(skips):.0f} '
			f'range [{min(skips)}, {max(skips)}]')

	# every jittered crop keeps an <eom> in its target half, because describe() drops the jitter rather
	# than the supervision when it does not. Measured 26% of halves on this corpus carry no <eom>.
	cancelled = sum(1 for i in on.indices
		if not on.describe(i)['jitter'] and not off.describe(i)['head'])
	print(f'  {cancelled} interior crops ended at offset 0 (drew 0, or the jitter was cancelled '
		'for want of an <eom>)')

	# and the mask the model actually sees
	items = [on._item(i) for i in on.indices[:8]]
	batch = on.collateBatch(items)
	ok_mask = True
	for row, (ids, sep, _, skip) in enumerate(items):
		mask = batch['target_mask'][row]
		if int(mask.sum()) != len(ids) - sep - 1 - skip or mask[:sep + 1 + skip].any() \
				or not mask[sep + 1 + skip:len(ids)].all():
			ok_mask = False
	check('collateBatch starts the mask at sep + 1 + skip', ok_mask)
	print(f'  supervised fractions: ' + ' '.join(
		f'{int(batch["target_mask"][r].sum()) / (len(i[0]) - i[1] - 1):.2f}'
		for r, i in enumerate(items)))


def length_table (root, samples, rng):
	'''9. What T actually comes out at, per pairing and line cap — attention is O(T^2), so the p99
	is what sizes a run, not the median. The last row of each block is the sampled range, which is what
	a config normally uses: its p99 sits near the fixed cap at the range's TOP, while its median falls
	well below, because the cap is drawn per crop.'''
	print('\n== 9. sequence length by pairing and line cap')
	print(f'  {"pairing":38s} {"line_range":>12s} {"median":>7s} {"p95":>7s} {"p99":>7s} {"max":>7s}')
	for source_dir, target_dir in PAIRINGS:
		for line_range in (128, 256, 512, (20, 256)):
			dataset = feeder(root, source_dir, target_dir, line_range=line_range)
			lengths = [len(case['ids']) for case in crops(dataset, samples, rng)
				if not case['degenerate']]
			lengths.sort()
			label = f'{source_dir} -> {target_dir}'
			shown = f'{line_range[0]}..{line_range[1]}' if isinstance(line_range, tuple) else str(line_range)
			print(f'  {label:38s} {shown:>12s} {statistics.median(lengths):7.0f} '
				f'{lengths[int(len(lengths) * .95)]:7d} {lengths[int(len(lengths) * .99)]:7d} '
				f'{lengths[-1]:7d}')


def check_transposition (root, source_dir, target_dir):
	'''13. transposition_sigma: one offset per sample, applied to both halves, nothing else disturbed.

	The property that matters is CORRESPONDENCE: source and target must move by the same number of
	semitones, or the pair no longer describes the same music and the supervision is a lie. So the walks
	below re-derive each half's note pitches and key signatures independently of the feeder and compare
	the two shifts.

	The augmentation moves two things in two places, and they are checked separately because they have
	different invariants:

	  note pitches    on the assembled ids. Length CANNOT change, so `_transpose_ids` is checked to
	                  leave every non-pitch id and the sequence length exactly as they were. A
	                  `control_change`/`program_change` `#XX` is a controller or program number, not a
	                  pitch, and must not move. Out of range folds by octaves, not by clamping —
	                  clamping would change the pitch class and put a wrong interval in the sample.
	  key signatures  on the text, walking the circle of fifths. Length CAN change, because `sf` is a
	                  signed byte in minimal hex (one nibble token when non-negative, two when
	                  negative). So the check is not that the length is fixed but that it changed by
	                  exactly the key-signature width delta, that positions stayed one-for-one with
	                  ids, and that the 'absolute' axis was corrected for it.
	'''
	print('\n== 13. transposition_sigma augmentation')
	kw = dict(mark_mode='tick', line_range=[20, 256], split='0/1', random_crop=False)
	off = feeder(root, source_dir, target_dir, **kw)
	check('default is off', off.transposition_sigma == 0.0, str(off.transposition_sigma))

	explicit = feeder(root, source_dir, target_dir, transposition_sigma=0.0, **kw)
	same = all(off.describe(i)['ids'] == explicit.describe(i)['ids'] for i in off.indices)
	check('sigma=0 is bit-identical to the default', same)
	check('the default reports transpose 0',
		all(off.describe(i)['transpose'] == 0 for i in off.indices))

	vocab = off.tokenizer.tokens

	def note_pitches (ids):
		'''(keyword, pitch) per note event, plus the non-note `#XX` tokens — walked here rather than
		read off the feeder, so a bug in _transpose_ids cannot hide behind its own helper.'''
		notes, others, current = [], [], None
		for tid in ids:
			token = vocab[tid]
			if token in ('note_on', 'note_off', 'control_change', 'program_change'):
				current = token
			elif token.startswith('#'):
				pitch = int(token[1:], 16)
				(notes if current in ('note_on', 'note_off') else others).append((current, pitch))
				current = None
			elif not (len(token) > 1 and token[0] in ('$', 'C')):
				current = None
		return notes, others

	def key_signatures (ids):
		'''(sf, mi tokens) per key_signature in an id stream, re-read from the tokens.'''
		tokens = [vocab[i] for i in ids]
		out = []
		for i, token in enumerate(tokens):
			if token != 'key_signature':
				continue
			j, nibbles = i + 1, []
			while j < len(tokens) and tokens[j] != '_':
				nibbles.append(tokens[j])
				j += 1
			if nibbles and j < len(tokens):
				out.append((S._decode_sf(nibbles), tokens[j + 1:j + 2]))
		return out

	# --- the pitch mechanism, at a fixed offset -------------------------------------------------
	lo, hi = off._pitch_lo, off._pitch_hi

	def expect (pitch, offset):
		want = pitch + offset
		while want < lo:
			want += 12
		while want > hi:
			want -= 12
		return want

	moved_bad = other_bad = struct_bad = 0
	for index in off.indices:
		case = off.describe(index)
		before, others0 = note_pitches(case['ids'])
		for offset in (-25, -12, -7, -1, 1, 5, 12, 24):
			shifted = off._transpose_ids(case['ids'], offset)
			after, others1 = note_pitches(shifted)
			if others0 != others1:
				other_bad += 1
			if len(shifted) != len(case['ids']) or not all(
					vocab[case['ids'][i]].startswith('#')
					for i, (a, b) in enumerate(zip(case['ids'], shifted)) if a != b):
				struct_bad += 1
			moved_bad += sum(1 for (_, a), (_, b) in zip(before, after) if b != expect(a, offset))
	check('every note pitch moves by the offset', moved_bad == 0, f'{moved_bad} wrong')
	check('a non-note `#XX` never moves', other_bad == 0, f'{other_bad} bad')
	check('_transpose_ids changes only pitch ids, never the length', struct_bad == 0, f'{struct_bad} bad')

	class_bad = range_bad = 0
	for index in off.indices[:8]:
		before, _ = note_pitches(off.describe(index)['ids'])
		for offset in (-60, -40, 40, 60):
			after, _ = note_pitches(off._transpose_ids(off.describe(index)['ids'], offset))
			for (_, a), (_, b) in zip(before, after):
				if (b - a - offset) % 12:
					class_bad += 1
				if not lo <= b <= hi:
					range_bad += 1
	check('a folded pitch keeps its pitch class', class_bad == 0, f'{class_bad} bad')
	check('a folded pitch stays in the vocab range', range_bad == 0, f'{range_bad} out of range')

	# --- the key-signature cycle ----------------------------------------------------------------
	cycle = list(S.KEY_SIGNATURE_CYCLE)
	check('the cycle is the circle of fifths, one semitone per place',
		all(S._fold_key(cycle[i] + 7) == cycle[(i + 1) % 12] for i in range(12)), str(cycle))
	check('the cycle covers exactly [-5, 6]', sorted(cycle) == list(range(-5, 7)))
	check('twelve semitones return every key to itself',
		all(S._fold_key(sf + 7 * 12) == sf for sf in cycle))
	check('the three enharmonic spellings fold as specified',
		(S._fold_key(7), S._fold_key(-7), S._fold_key(-6)) == (-5, 5, 6),
		f'7->{S._fold_key(7)} -7->{S._fold_key(-7)} -6->{S._fold_key(-6)}')
	check('the sf codec round-trips [-7, 7]',
		all(S._decode_sf(S._encode_sf(sf)) == sf for sf in range(-7, 8)))
	# the width change is the whole reason this runs on text rather than on ids
	widths = {sf: len(S._encode_sf(sf)) for sf in range(-6, 8)}
	check('a negative sf takes two nibble tokens and a non-negative one takes one',
		all(w == (2 if sf < 0 else 1) for sf, w in widths.items()), str(widths))

	# --- the option end to end ------------------------------------------------------------------
	sigma = 4.0
	on = feeder(root, source_dir, target_dir, transposition_sigma=sigma, **kw)
	offsets, pair_bad, key_bad, key_range_bad, mode_bad = [], 0, 0, 0, 0
	shape_bad, len_bad, moved_keys = 0, 0, 0
	for index in on.indices:
		plain, moved = off.describe(index), on.describe(index)
		offset = moved['transpose']
		offsets.append(offset)

		# positions stay one-for-one with ids, and <sep> still points at <sep>
		if len(moved['positions']) != len(moved['ids']) \
				or moved['ids'][moved['sep']] != on.tokenizer.sep_id:
			shape_bad += 1

		# the length may change, but by EXACTLY the key-signature width delta
		delta = (on._key_delta(moved['source'], offset, plain['source_range'][1])
			- on._key_delta(moved['source'], offset, plain['source_range'][0])
			+ on._key_delta(moved['target'], offset, plain['target_range'][1])
			- on._key_delta(moved['target'], offset, plain['target_range'][0]))
		if len(moved['ids']) - len(plain['ids']) != delta:
			len_bad += 1

		# pitches: both halves by the same offset
		src0, _ = note_pitches(plain['ids'][:plain['sep']])
		tgt0, _ = note_pitches(plain['ids'][plain['sep'] + 1:])
		src1, _ = note_pitches(moved['ids'][:moved['sep']])
		tgt1, _ = note_pitches(moved['ids'][moved['sep'] + 1:])
		shifts_src = {b - a for (_, a), (_, b) in zip(src0, src1)}
		shifts_tgt = {b - a for (_, a), (_, b) in zip(tgt0, tgt1)}
		if shifts_src - {offset} or shifts_tgt - {offset} \
				or (src0 and tgt0 and shifts_src != shifts_tgt):
			pair_bad += 1

		# key signatures: the cycle, the range, and the untouched mode flag
		keys0, keys1 = key_signatures(plain['ids']), key_signatures(moved['ids'])
		if len(keys0) != len(keys1):
			key_bad += 1
			continue
		if offset and keys0 != keys1:
			moved_keys += 1
		for (sf0, mi0), (sf1, mi1) in zip(keys0, keys1):
			if sf1 != S._fold_key(sf0 + 7 * offset):
				key_bad += 1
			if offset and not -5 <= sf1 <= 6:
				key_range_bad += 1
			if mi0 != mi1:
				mode_bad += 1

	check('<sep> and the position/id lengths stay consistent', shape_bad == 0, f'{shape_bad} bad')
	check('the length changes by exactly the key-signature width delta', len_bad == 0, f'{len_bad} bad')
	check('source and target pitches shift by the SAME offset', pair_bad == 0, f'{pair_bad} bad')
	check('every key signature walks the cycle', key_bad == 0, f'{key_bad} bad')
	check('a transposed key signature lands in [-5, 6]', key_range_bad == 0, f'{key_range_bad} bad')
	check('the major/minor flag is never touched', mode_bad == 0, f'{mode_bad} bad')
	check('key signatures are actually moved', moved_keys > 0, f'{moved_keys} crops')
	check('offsets are actually applied', sum(1 for o in offsets if o) > len(offsets) * 0.5,
		f'{sum(1 for o in offsets if o)}/{len(offsets)} nonzero')
	check('every offset is an integer', all(isinstance(o, int) for o in offsets))
	mean, spread = statistics.mean(offsets), statistics.pstdev(offsets)
	check('offsets center on no transposition', abs(mean) < sigma * 0.5, f'mean {mean:+.2f}')
	check('offset spread matches the requested sigma', abs(spread - sigma) < sigma * 0.5,
		f'std {spread:.2f} vs {sigma}')
	print(f'  offsets: n {len(offsets)} mean {mean:+.2f} std {spread:.2f} '
		f'range [{min(offsets)}, {max(offsets)}]')

	# --- the 'absolute' axis survives the width change ------------------------------------------
	# 'absolute' places the source half against the END of its file, so a key signature that changed
	# width has to be corrected for or every crop sits a token off. A tail crop is where that shows:
	# its last source token is the file's last, which must land on exactly -2.
	abs_on = feeder(root, source_dir, target_dir, transposition_sigma=sigma,
		**dict(kw, pos_style='absolute'))
	tails = [i for i in abs_on.indices if abs_on.describe(i)['tail']]
	anchored = sign_bad = 0
	for index in abs_on.indices:
		case = abs_on.describe(index)
		src = case['positions'][:case['sep']]
		tgt = case['positions'][case['sep'] + 1:]
		if (src and max(src) > -2) or (tgt and min(tgt) < -1) \
				or any(b - a != 1 for a, b in zip(src, src[1:])):
			sign_bad += 1
		if case['tail'] and case['sep'] and case['positions'][case['sep'] - 1] == -2:
			anchored += 1
	check("'absolute' keeps the halves apart and each half contiguous", sign_bad == 0, f'{sign_bad} bad')
	check("'absolute' tail crops still land on -2", tails and anchored == len(tails),
		f'{anchored}/{len(tails)}')

	again = feeder(root, source_dir, target_dir, transposition_sigma=sigma, **kw)
	check('a deterministic crop transposes the same way twice',
		all(again.describe(i)['ids'] == on.describe(i)['ids'] for i in on.indices))

	try:
		feeder(root, source_dir, target_dir, transposition_sigma=-1.0, **kw)
		check('a negative sigma is rejected', False, 'no error')
	except ValueError:
		check('a negative sigma is rejected', True)


def check_pos_style (root, source_dir, target_dir, samples):
	'''11. The three pos_style conventions.

	'flat' and 'sep' must produce IDENTICAL ids and be equivalent to a RoPE model (one arithmetic run,
	and a uniform shift is invisible to RoPE) — 'sep' buys readability, not behaviour. 'absolute' must
	place the source at <= -2 and the target at >= -1, which is the property that makes it collision-
	free and order-preserving no matter how long the files are.
	'''
	print('\n== 11. pos_style')
	sets = {}
	for style in ('flat', 'sep', 'absolute'):
		sets[style] = feeder(root, source_dir, target_dir, mark_mode='tick',
			pos_style=style, random_crop=False)

	check('bad pos_style raises', _raises(lambda: feeder(root, source_dir, target_dir,
		pos_style='bogus')), 'a typo must not fall through to a default')

	ok_len = ok_ids = ok_flat = ok_sep = ok_abs_sign = ok_abs_sep = ok_cross = True
	seen_head = seen_tail = seen_mid = 0
	for i in range(min(samples, len(sets['flat']))):
		index = sets['flat'].indices[i % len(sets['flat'])]
		cases = {style: ds.describe(index) for style, ds in sets.items()}
		for style, case in cases.items():
			if len(case['positions']) != len(case['ids']):
				ok_len = False
		# the ids must not depend on the position convention at all
		if not (cases['flat']['ids'] == cases['sep']['ids'] == cases['absolute']['ids']):
			ok_ids = False

		flat, sep_c, absol = cases['flat'], cases['sep'], cases['absolute']
		n, s = len(flat['ids']), flat['sep']
		if flat['positions'] != list(range(n)):
			ok_flat = False
		# 'sep': one run through -2, -1, 0 -- i.e. flat shifted so <sep> lands on -1
		if sep_c['positions'] != list(range(-(s + 1), n - s - 1)):
			ok_sep = False
		if sep_c['positions'][s] != -1:
			ok_sep = False

		pos = absol['positions']
		s = absol['sep']
		if max(pos[:s]) > -2 or pos[s] != -1 or min(pos[s + 1:]) < -1:
			ok_abs_sign = False
		# no CONTENT token of one half may share a position with the other's
		if max(pos[:s]) >= min(x for x in pos[s + 1:] if x >= 0):
			ok_cross = False
		# a tail crop ends the source at exactly -2; a head crop starts the target's content at 0
		if absol['tail']:
			seen_tail += 1
			if pos[s - 1] != -2:
				ok_abs_sep = False
		elif absol['head']:
			seen_head += 1
		else:
			seen_mid += 1

	check('positions length == ids length (all styles)', ok_len)
	check('ids independent of pos_style', ok_ids)
	check("'flat' == 0..T-1", ok_flat)
	check("'sep' is one run with <sep> at -1", ok_sep)
	check("'absolute' source <= -2, <sep> == -1, target >= -1", ok_abs_sign)
	check("'absolute' tail crop ends source at -2", ok_abs_sep, f'{seen_tail} tail crops seen')
	check("'absolute' halves never share a content position", ok_cross)
	check('head/tail/middle all covered', seen_head and seen_tail and seen_mid,
		f'head {seen_head} tail {seen_tail} middle {seen_mid}')

	# 'absolute' must actually MOVE with the crop -- otherwise it is just 'sep' under another name.
	spans = set()
	for i in range(min(40, len(sets['absolute']))):
		case = sets['absolute'].describe(sets['absolute'].indices[i])
		spans.add(case['positions'][case['sep'] + 1])
	check("'absolute' target start varies by crop", len(spans) > 1, f'{len(spans)} distinct starts')

	# collateBatch must carry positions and continue each row's run into the padding, or a padded row
	# stops matching its unpadded self (measured 4e-2 on the hidden states before this was fixed).
	ds = sets['absolute']
	items = [ds[i] for i in range(min(4, len(ds)))]
	batch = ds.collateBatch(items)
	width = batch['input_ids'].shape[1]
	ok_pad_pos = True
	for row, (ids, _, positions, _) in enumerate(items):
		if not torch.equal(batch['position_ids'][row, :len(ids)], positions):
			ok_pad_pos = False
		for k in range(len(ids), width):		# the pad tail continues the run, +1 per slot
			if int(batch['position_ids'][row, k]) != int(positions[-1]) + (k - len(ids) + 1):
				ok_pad_pos = False
	check('collateBatch position_ids: real slots verbatim, pad continues the run', ok_pad_pos)


def _raises (fn):
	try:
		fn()
		return False
	except ValueError:
		return True


def main ():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--root', default=DEFAULT_ROOT)
	parser.add_argument('--samples', type=int, default=300)
	args = parser.parse_args()

	rng = random.Random(20260811)

	for source_dir, target_dir in PAIRINGS:
		for mark_mode in ('measure', 'tick'):
			print(f'\n{"=" * 78}\n{source_dir} -> {target_dir}  [mark_mode={mark_mode}]\n{"=" * 78}')
			dataset = feeder(args.root, source_dir, target_dir, mark_mode=mark_mode)
			if source_dir == PAIRINGS[0][0] and mark_mode == 'measure':
				check_vocab(dataset)
			check_crops(dataset, args.samples, rng)
			check_head_tail(dataset, 2000, rng)
			check_tail_degrades(dataset, 2000, rng)
			check_collate(dataset)
			check_describe(dataset)

	check_determinism(args.root, *PAIRINGS[0])
	check_line_range(args.root, *PAIRINGS[2], args.samples, rng)
	check_pos_style(args.root, *PAIRINGS[0], args.samples)
	check_start_jitter(args.root, *PAIRINGS[2])
	check_transposition(args.root, *PAIRINGS[2])
	length_table(args.root, 200, rng)

	print(f'\n{"=" * 78}')
	if failures:
		print(f'FAILED ({len(failures)}):')
		for line in failures:
			print(f'  - {line}')
		sys.exit(1)
	print('seq2seq2 feeder: all checks ok')


if __name__ == '__main__':
	main()

