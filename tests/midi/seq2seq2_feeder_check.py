
'''Checks for starry.midi.data.seq2seq2.Seq2Seq2 over the three-arm test202608 corpus.

Run: python tests/midi/seq2seq2_feeder_check.py [--root DIR] [--samples N]

The feeder's whole job is that a crop is MARK-ALIGNED: the source window starts and ends on a
directive naming a score position, and the target window covers the same music because it was found
by that directive's key. So the checks here are mostly about boundaries and losslessness, not about
tensor plumbing:

  1. vocab layout   — <sep> is id 5, vocab_size 838, no other special moved
  2. round-trip     — decoding the source half reproduces the cropped source lines exactly
  3. no <unknown>   — the corpus is fully in-vocabulary, so a miss means an encoder bug
  4. boundaries     — ranges land on marks; <bos> iff the crop starts at the piece start; <eos> ends
                      every target exactly once and never appears in the source; <eom> count
  5. alignment      — boundary keys exist in both files; outward-walk distance reported
  6. head/tail rate — the configured p_head/p_tail actually come out
  7. determinism    — random_crop=False repeats exactly; splits are disjoint and stable
  8. collateBatch   — shapes, padding, target_mask covering exactly the post-<sep> region
  8b. describe()    — same crop as __getitem__; one <eom> per measure it names (the vis relies on it)
  9. length table   — median/p95/p99/max of T per pairing and line cap, so a config can be sized
 10. line_range    — the per-crop cap varies, honours both ends, and stays deterministic
 11. pos_style     — flat/sep/absolute: ids unchanged, <sep> anchoring, sign separation, pad run
'''

import argparse
import os
import random
import statistics
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from starry.midi.data.seq2seq2 import Seq2Seq2, _get_file	# noqa: E402


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
	check('vocab_size == 838', t.vocab_size == 838, f'got {t.vocab_size}')
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
		source = _get_file(os.path.join(dataset.source_root, name), dataset.mark_mode)
		target = _get_file(os.path.join(dataset.target_root, name), dataset.mark_mode)
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
	source = _get_file(os.path.join(dataset.source_root, dataset.names[index]), dataset.mark_mode)
	head = tail = 0
	for _ in range(samples):
		a, z = dataset._pick_crop(source, rng)
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
		ids, sep, positions = dataset[i]
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
	width = max(len(ids) for ids, _, _ in items)
	check('keys', set(batch) == {'input_ids', 'masks', 'target_mask', 'sep_index', 'position_ids'},
		str(set(batch)))
	check('input_ids shape', tuple(batch['input_ids'].shape) == (len(items), width),
		str(tuple(batch['input_ids'].shape)))
	check('all long dtype', all(v.dtype == torch.long for v in batch.values()))
	ok_pad = ok_mask = ok_target = ok_sep = True
	for row, (ids, sep, positions) in enumerate(items):
		length = len(ids)
		if length < width and not (batch['input_ids'][row, length:] == t.pad_id).all():
			ok_pad = False
		if batch['masks'][row].sum().item() != length:
			ok_mask = False
		# the supervised region is exactly (sep, length)
		if batch['target_mask'][row].sum().item() != length - sep - 1:
			ok_target = False
		if batch['target_mask'][row, :sep + 1].any():
			ok_target = False
		if batch['input_ids'][row, sep].item() != t.sep_id or batch['sep_index'][row].item() != sep:
			ok_sep = False
	check('right-padded with <pad>', ok_pad)
	check('masks count real tokens', ok_mask)
	check('target_mask covers exactly the post-<sep> region', ok_target)
	check('sep_index points at <sep>', ok_sep)


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
		source = _get_file(os.path.join(dataset.source_root, dataset.names[index]), dataset.mark_mode)
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
	for row, (ids, _, positions) in enumerate(items):
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
			check_collate(dataset)
			check_describe(dataset)

	check_determinism(args.root, *PAIRINGS[0])
	check_line_range(args.root, *PAIRINGS[2], args.samples, rng)
	check_pos_style(args.root, *PAIRINGS[0], args.samples)
	check_start_jitter(args.root, *PAIRINGS[2])
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

