'''Does `starry/midi/align.py` recover the RIGHT correspondence on real score<->irregular pairs?

`tests/midi/align_check.py` already verifies the parts that are decidable: the elapse automaton, the
feasibility mask against brute force, the grammar walk, and AlignState's cost/prior bounds on
SYNTHETIC sources. None of that answers the question this file asks, because a mask can be perfectly
correct combinatorially while the alignment it is derived from points at the wrong notes. What was
missing is the measurement on real data: given the irregular arm as source and the score arm as
target, does `AlignState.observe` pair a generated note with the source note it actually came from?

GROUND TRUTH, and why `@measure` is the coordinate. The two arms are the same piece -- the irregular
arm IS the score arm run through pianistMockerMIDI -- so a correspondence exists by construction. But
it is not recoverable from the token streams by identity:

  - note ORDER differs. Measured over these 100 files, the longest common pitch prefix is 0-4 notes:
    a chord's notes may be emitted in any order, and the mocker reorders them.
  - note COUNT differs. The mocker DROPS notes (median -14 per file over these 100) and occasionally
    re-pitches one. Only 5 of 100 files have equal pitch multisets, so a positional pairing is wrong
    on 95 of them.
  - absolute TICK differs, by design: the source carries rubato and the target is quantised.

What survives is the DIRECTIVES, and there are two of them. `starry/midi/data/seq2seq2.py` says so
for the feeder's own crop alignment ("@measure keys are present in both arms for every mark (0 missing
of 8097), while @tick keys go missing 0.12% ... and 3.08%"), and both hold up when re-measured here:

  @measure   identical, contiguous 1..N sequence in both arms on all 100 files.
  @tick      MEASURE-RELATIVE (it resets at every @measure), so the key is the PAIR `(measure, tick)`
             exactly as seq2seq2 states. Keyed that way: 49,188 shared keys over these 100 files, with
             the pitch multiset equal on 0.941 of them, 0.51% of score keys missing from the irregular
             arm and 4.54% the other way. The DIRECTIVE VALUE is identical in both arms while the real
             onsets diverge (score quantised 0,240,480...; irregular 0,201,411... on the same bar) --
             which is precisely what makes it ground truth: it records the ORIGINAL score position and
             survives the mocker's perturbation of the actual timing.

An earlier version of this file rejected @tick on the grounds that its per-file count differs between
arms (638 vs 636, 217 vs 441). That was a measurement BUG, not a property of the data: keying on tick
alone collides every measure's tick-0 group, which collapsed 49,188 real groups into 1,214 and dropped
apparent agreement to 0.514. The lesson generalises past this file -- a coordinate that looks unusable
is worth re-checking against the definition before it is discarded, because discarding it silently
costs the STRICTEST check available.

Three grades of ground truth, deliberately kept apart:

  TICK (hard, fine)   a matched pair must land in the same `(measure, tick)` group. ~1-3 notes per
                      group against a bar's ~16, so this is the strictest criterion here, and it
                      applies to 0.995 of matches.
  MEASURE (hard)      same measure index. Coarser, and kept precisely BECAUSE it is coarser: it is the
                      criterion the feeder's own crop alignment relies on, so a regression that shows
                      up here is a regression in something production depends on.
  DP PAIR (soft)      Needleman-Wunsch on pitch within a shared measure. A SECOND ALIGNER, not truth.
                      Now that @tick is available it is also MEASURABLE: the DP reference disagrees
                      with @tick on 7.0% of its own pairings, so 7 points of align.py's apparent
                      "error" against it are the reference's, not align.py's.

THE DECISIVE CHECK IS THE WRONG-PAIR CONTROL. Every "recovery" number is meaningless without it: an
aligner that matches promiscuously scores well on measure agreement simply because a wrong note is
often in the right bar. So the same target is also aligned against a DIFFERENT piece's source, and
the true pairing must beat it. MEASURED: matchrate 0.980 vs 0.028, winning on 100/100 files. That gap
is what makes the rest evidence rather than decoration. It widened sharply (from 0.935 vs 0.205) when
the lattice landed, because the decoy's collapse is now a REFUSAL to match rather than a run of
confident wrong pairings -- the same property that stopped whole files from being poisoned.

Note what matchrate is and is NOT: it is `matched / (matched + misses)`, i.e. COVERAGE -- whether a
pairing was made at all -- not whether the pairing was RIGHT. The correctness figures are the @tick
ones below (precision 0.9503, recall 0.9337). Reading matchrate as accuracy is what makes a cross-tool
comparison meaningless, since the music-widgets Matcher's headline coverage measures the same
coverage-only quantity.

Honest limits of what this file establishes:

  - `tick_interval`'s mask contains the true target tick on 0.595 of the notes it fires on (0.646 on
    the 91 files that align well). It is the WEAKEST result here and it is reported rather than
    tuned away. Two things bound how much it condemns: the missing third is partly DP-reference
    error, not necessarily mask error; and `tick_interval` currently has NO PRODUCTION CALLER (grep:
    only align.py and the two test files), so this is a latent property, not a live mis-kill.
  - Thresholds below are set from the MEASURED distribution over all 100 files, with margin at p10,
    not from a target. `--samples` below 100 is noisier than the defaults assume: the first 14 files
    in sorted order happen to be a weak subset (measure-agree 0.75 against the corpus's 0.89), so a
    small run may fail on sampling alone. Prefer the default.

Run:  python tests/midi/align_corpus_check.py [--root DIR] [--samples N] [--src-window T]
'''

import argparse
import bisect
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.midi.align import AlignState, soft_indices
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer

sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'midi'))
from translateMidiseq2 import encode_lines, keyword_tokens, note_on_events


DEFAULT_ROOT = os.path.expanduser('~/data/midi/test202608')
# The pair direction is the training one: configs/midi-translator-nota1m-*.yaml states the arms as
# `midi-seq2-irregular` -> `midi-seq2-score`, i.e. the PERFORMANCE conditions the generation of the
# SCORE. Aligning them the other way round would be a different (and untested) claim.
SRC_ARM = 'midiseq2-irregular'
TGT_ARM = 'midiseq2-score'
# Fallback names: the corpus root uses `midi-seq2-*` while the nota1m-100 subset uses `midiseq2-*`.
ARM_ALIASES = {'midiseq2-irregular': 'midi-seq2-irregular', 'midiseq2-score': 'midi-seq2-score'}

# Production default from translateMidiseq2Beam.py's tuning record (src_window 960 / prime_window
# 320 -> onsetF1 0.514). The regime matters: AlignState.__init__ says whole-file alignment has no
# bound on where a note may match, since every pitch recurs dozens of times across a piece, whereas
# "the source WINDOW is itself the bound". MEASURED, and it is not a small effect -- whole-file
# alignment scores measure-agree 0.44 against the windowed 0.89 on the same files.
SRC_WINDOW = 960

# The sample count every threshold below was measured at. Stated as a constant because the thresholds
# are only meaningful relative to it.
CALIBRATED_SAMPLES = 100

# Minimum notes per side for a file to carry signal at all. Below this the anchor vote never gets
# enough pairs to form (MinPairsForMask is 4) and the file measures the cold start, not the aligner.
MIN_NOTES = 8


def resolve_dir (root, arm):
	'''The corpus directory for `arm`, trying the nota1m-100 subset then the root, both spellings.'''
	for base in (os.path.join(root, 'nota1m-100'), root):
		for name in (arm, ARM_ALIASES.get(arm, arm)):
			path = os.path.join(base, name)
			if os.path.isdir(path):
				return path
	return None


def load_notes (path, limit_tokens=None):
	'''One arm's note_on events, each carrying measure index and softIndex.

	Everything here goes through the PRODUCTION walk (`encode_lines` + `note_on_events`) rather than a
	local re-parse, so the test cannot pass against a reimplementation that the real pipeline does not
	share. `eom=True` is what makes measure attribution possible at all: @measure lines are control,
	not content, and only the <eom> form leaves a token to count.

	Measure index is 1 + (number of <eom> marks before the note's pitch token), which matches the text:
	`@measure 1` emits nothing (it opens the piece rather than closing a bar), so everything before the
	first mark is measure 1.
	'''
	tokenizer = load_notes.tokenizer
	keywords = load_notes.keywords
	lines = [l for l in open(path).read().split('\n') if l.strip()]
	marks = []
	ids = encode_lines(lines, tokenizer, True)
	if limit_tokens:
		ids = ids[:limit_tokens]
	events, _abst, _state = note_on_events(ids, tokenizer, keywords, marks=marks)
	mark_at = [m[0] for m in marks]
	for e in events:
		e['measure'] = 1 + bisect.bisect_right(mark_at, e['pitch_index'])
	for e, si in zip(events, soft_indices([e['onset'] for e in events])):
		e['softIndex'] = si
	return events


def note_keys (path):
	"""The `(measure, tick)` key each note_on sits under, in note order, read from TEXT.

	The key is the PAIR because @tick is MEASURE-RELATIVE -- it resets at every @measure line, so tick
	alone collides every bar's tick-0 group. That collision is not hypothetical: it is the bug that
	made an earlier version of this file discard @tick as unusable (1,214 apparent groups against the
	real 49,188, agreement 0.514 against the real 0.941).

	`t is None` marks a note with no @tick above it in its bar, which the callers must skip rather than
	group under a fabricated key. Read from text for the same reason as `measure_sequence`: this is
	ground truth, so it comes from the directive rather than from anything the token walk produced.
	"""
	out = []
	measure, tick = 1, None
	for line in open(path):
		line = line.strip()
		if not line:
			continue
		if line.startswith('@measure'):
			measure, tick = int(line.split()[1]), None
			continue
		if line.startswith('@tick'):
			tick = int(line.split()[1])
			continue
		fields = line.split()
		if fields[0] == 'note_on' and any(f.startswith('#') for f in fields):
			out.append((measure, tick))
	return out


def measure_sequence (path):
	'''The raw `@measure N` numbers, read from TEXT.

	Deliberately not via the token walk: this is the ground-truth coordinate, so it is read from the
	directive itself rather than from a count of <eom> tokens that the walk produced. If the two ever
	disagree the walk is what is wrong, and a ground truth derived from the walk could not say so.
	'''
	return [int(l.split()[1]) for l in open(path) if l.startswith('@measure')]


def dp_pairs (src, tgt):
	'''tgt index -> src index, by Needleman-Wunsch on pitch WITHIN each shared measure.

	A SECOND ALIGNER, not ground truth -- see the module docstring. Restricted to one measure at a
	time for two reasons: it keeps the cost quadratic in a bar (~16 notes) instead of in the file, and
	more importantly it cannot invent a cross-bar pairing that the hard measure coordinate would
	reject. A mismatched pitch costs 10 against a gap's 1, so the walk prefers to DROP a note rather
	than pair it with a different pitch -- which is the right bias here, because the mocker's edit is
	overwhelmingly a drop (measured: `only-score` non-empty on nearly every differing bar, while
	`only-irregular` is usually empty).
	'''
	out = {}
	src_by_measure, tgt_by_measure = {}, {}
	for i, e in enumerate(src):
		src_by_measure.setdefault(e['measure'], []).append(i)
	for i, e in enumerate(tgt):
		tgt_by_measure.setdefault(e['measure'], []).append(i)
	for measure in set(src_by_measure) & set(tgt_by_measure):
		si, ti = src_by_measure[measure], tgt_by_measure[measure]
		n, k = len(ti), len(si)
		cost = [[0] * (k + 1) for _ in range(n + 1)]
		for a in range(1, n + 1):
			cost[a][0] = a
		for b in range(1, k + 1):
			cost[0][b] = b
		for a in range(1, n + 1):
			for b in range(1, k + 1):
				same = tgt[ti[a - 1]]['pitch'] == src[si[b - 1]]['pitch']
				cost[a][b] = min(cost[a - 1][b - 1] + (0 if same else 10),
					cost[a - 1][b] + 1, cost[a][b - 1] + 1)
		a, b = n, k
		while a > 0 and b > 0:
			same = tgt[ti[a - 1]]['pitch'] == src[si[b - 1]]['pitch']
			if cost[a][b] == cost[a - 1][b - 1] + (0 if same else 10):
				if same:
					out[ti[a - 1]] = si[b - 1]
				a -= 1
				b -= 1
			elif cost[a][b] == cost[a - 1][b] + 1:
				a -= 1
			else:
				b -= 1
	return out


def pair_files (src_dir, tgt_dir, name, src_window):
	'''(src, tgt) note lists for one piece, or (None, None) when the file carries too little signal.

	The source is cropped to `src_window` TOKENS -- the production unit, not a note count -- and the
	target is then cropped to the measures that crop fully covers. Without that second crop the target
	would run past the end of the source window and every note beyond it would be a forced miss,
	which measures the crop rather than the aligner.
	'''
	src = load_notes(os.path.join(src_dir, name), src_window)
	if not src:
		return None, None
	last_full = max(e['measure'] for e in src) - 1
	tgt = [e for e in load_notes(os.path.join(tgt_dir, name)) if e['measure'] <= last_full]
	if len(src) < MIN_NOTES or len(tgt) < MIN_NOTES:
		return None, None
	# The (measure, tick) key rides along on the note list. zip() against the FULL text walk is safe
	# for the target (it was cropped by measure, so it is a prefix) and for the source (cropped by
	# token count, also a prefix) -- both crops keep note order, so the nth key belongs to the nth note.
	for e, key in zip(src, note_keys(os.path.join(src_dir, name))):
		e['key'] = key
	for e, key in zip(tgt, note_keys(os.path.join(tgt_dir, name))):
		e['key'] = key
	return src, tgt


def align_pair (src, tgt, ref=None):
	'''Fold every target note through AlignState, collecting the per-note verdicts.

	`tick_interval` is queried BEFORE `observe` and on the GROUND-TRUTH source note's onset, which is
	the contract its docstring states ("Feasible absolute target tick interval for a note aligned to
	`src_tick`") and what align_check.py's own callers pass. Querying it on the target tick instead is
	a category error that silently reads as a broken mask -- measured at 0.09 contains-true, against
	0.60 once the argument is right.
	'''
	state = AlignState(src, seed_offset=0.0)
	measure_hit = matched = exact = ref_n = mask_hit = mask_n = backward = 0
	tick_hit = tick_n = tick_scorable = 0
	roll_hit = 0
	# (measure, tick) -> the source indices in that group. A GROUP, not a single index: a chord shares
	# one key, and its notes may be paired in any order, so demanding a specific index inside the group
	# would report the chords as errors exactly as an index-order charge would.
	src_groups = {}
	for i, e in enumerate(src):
		key = e.get('key')
		if key and key[1] is not None:
			src_groups.setdefault(key, set()).add(i)
	# ROLLED CHORDS. The mocker does not keep a chord simultaneous: the score's (1,480)[57,60,65,69,72]
	# becomes (1,480)[57,60] (1,543)[65] (1,607)[69] (1,672)[72] in the irregular arm -- a pianist
	# rolling the chord. Those new keys are NOT quantised (the irregular arm has 597 non-multiple-of-10
	# @tick values against the score arm's 0), and MEASURED corpus-wide, 90.6% of the notes sitting in
	# an irregular-only key are the same pitch in the same bar as a score note: MOVED by the mocker,
	# not added. So a strict key match charges align.py for finding the musically right note.
	#
	# The tolerant group admits a same-pitch source note in the same bar whose key the SCORE ARM DOES
	# NOT HAVE. That condition is what keeps it from being a licence: an irregular-only key cannot be
	# any other target note's truth, so admitting it creates no ambiguity about who owns the note.
	#
	# MEASURED effect, and it is much smaller than the 90.6% suggests: precision 0.7486 -> 0.7529,
	# recall 0.5904 -> 0.5922. A 960-token window spans 7-10 bars of a file, and a rolled note only
	# produces a wrong verdict when align.py actually matches TO it -- usually it matches right or
	# misses. Both numbers are reported because the gap between them is the ground truth's own slack,
	# and hiding it would make the strict figure look more exact than it is.
	tgt_keys = {e['key'] for e in tgt if e.get('key') and e['key'][1] is not None}
	rolled = {}
	for i, e in enumerate(src):
		key = e.get('key')
		if key and key[1] is not None and key not in tgt_keys:
			rolled.setdefault((key[0], e['pitch']), set()).add(i)
	dp_hit = dp_n = 0
	if ref:
		for j, i in ref.items():
			key = tgt[j].get('key')
			if key and key[1] is not None and key in src_groups:
				dp_n += 1
				dp_hit += i in src_groups[key]
	offsets = []
	prev_src = None
	for j, e in enumerate(tgt):
		gt = None if ref is None else ref.get(j)
		if gt is not None:
			lo, hi = state.tick_interval(src[gt]['onset'])
			if lo is not None:
				mask_n += 1
				mask_hit += lo <= e['onset'] <= hi
		# Counted BEFORE observe and regardless of whether it matches, because this is the denominator
		# of the RECALL figure: a target note that align.py never paired is a note it failed to
		# recover, and leaving it out (as tick_agree's denominator does) flatters the result.
		key = e.get('key')
		scorable = bool(key and key[1] is not None and key in src_groups)
		tick_scorable += scorable
		detail = state.observe(e['pitch'], e['onset'], e['softIndex'])
		index = detail['src']
		if index is None:
			continue
		matched += 1
		measure_hit += src[index]['measure'] == e['measure']
		if scorable:
			tick_n += 1
			tick_hit += index in src_groups[key]
			roll_hit += (index in src_groups[key]
				or index in rolled.get((key[0], e['pitch']), ()))
		offsets.append(src[index]['softIndex'] - e['softIndex'])
		# A drop of more than a chord's worth of source notes. Not >0: a LOWER index is normal inside
		# a chord (align.py's ReuseCost note measures 7 of 8 non-advancing steps as within-chord), so
		# charging every one of those would report the chords as errors.
		if prev_src is not None and index < prev_src - 4:
			backward += 1
		prev_src = index
		if gt is not None:
			ref_n += 1
			exact += index == gt
	total = matched + state.misses
	return dict(matched=matched, total=total,
		agree=measure_hit / matched if matched else 0.0,
		tick_agree=tick_hit / tick_n if tick_n else None,
		tick_recall=tick_hit / tick_scorable if tick_scorable else None,
		roll_agree=roll_hit / tick_n if tick_n else None,
		tick_cover=tick_n / matched if matched else 0.0,
		dp_quality=dp_hit / dp_n if dp_n else None,
		exact=exact / ref_n if ref_n else None,
		mask=mask_hit / mask_n if mask_n else None, mask_n=mask_n,
		matchrate=state.matched / total if total else 0.0,
		backward=backward / matched if matched else 0.0,
		prior=state.prior, ratio=state.ratio,
		offset_spread=(statistics.pstdev(offsets) if len(offsets) > 2 else 0.0))


def collect (src_dir, tgt_dir, names, src_window):
	'''Run every pairing once and cache the results, so the checks below share one pass.'''
	out = []
	for name in names:
		src, tgt = pair_files(src_dir, tgt_dir, name, src_window)
		if src is None:
			continue
		row = align_pair(src, tgt, dp_pairs(src, tgt))
		row['name'] = name
		row['src'] = src
		row['tgt'] = tgt
		out.append(row)
	return out


def pct (rows, key):
	'''(mean, median, p10) of a field, skipping files where it is undefined.'''
	vals = sorted(r[key] for r in rows if r.get(key) is not None)
	if not vals:
		return None, None, None
	return statistics.mean(vals), statistics.median(vals), vals[len(vals) // 10]


# --- checks -------------------------------------------------------------------------------

def check_ground_truth (src_dir, tgt_dir, names):
	'''The corpus assumption itself: @measure must be an identical, contiguous coordinate in both arms.

	FIRST because every other check is void without it. If the two arms disagree about bar numbering
	then "same measure" is not a correctness criterion and the recovery numbers below mean nothing --
	so this is a precondition, not a property of align.py.
	'''
	identical = contiguous = 0
	for name in names:
		a = measure_sequence(os.path.join(src_dir, name))
		b = measure_sequence(os.path.join(tgt_dir, name))
		identical += a == b
		contiguous += a == list(range(1, len(a) + 1))
	ok = identical == len(names) and contiguous == len(names)
	if not ok:
		print(f'  FAIL ground truth: {identical}/{len(names)} identical, '
			f'{contiguous}/{len(names)} contiguous')
	print(f'{"ok  " if ok else "FAIL"} ground truth: @measure identical in both arms on '
		f'{identical}/{len(names)} files, contiguous 1..N on {contiguous}/{len(names)}')
	return ok


def check_streams_differ (rows):
	'''The task must be NON-TRIVIAL: the two arms must not be recoverable by position or identity.

	Stated as a check rather than a comment because it is the reason `observe` is needed at all. If
	pitch sequences matched positionally, a zip() would align these files and every number below would
	be measuring nothing.
	'''
	equal_multiset = 0
	long_prefix = 0
	for r in rows:
		sp = sorted(e['pitch'] for e in r['src'])
		tp = sorted(e['pitch'] for e in r['tgt'])
		equal_multiset += sp == tp
		a = [e['pitch'] for e in r['src']]
		b = [e['pitch'] for e in r['tgt']]
		n = 0
		while n < min(len(a), len(b)) and a[n] == b[n]:
			n += 1
		long_prefix += n > 8
	ok = equal_multiset < len(rows) * 0.2 and long_prefix < len(rows) * 0.2
	if not ok:
		print(f'  FAIL triviality: {equal_multiset} equal multisets, {long_prefix} long prefixes')
	print(f'{"ok  " if ok else "FAIL"} task is non-trivial: equal pitch multiset on '
		f'{equal_multiset}/{len(rows)} windows, common prefix > 8 notes on {long_prefix}/{len(rows)}')
	return ok


def check_wrong_pair_control (src_dir, tgt_dir, names, src_window):
	'''THE DECISIVE CHECK. A true pairing must beat the same target against a DIFFERENT piece's source.

	Without this every recovery number is unfalsifiable: an aligner that matches promiscuously scores
	well on measure agreement, because a wrong note is frequently in the right bar anyway. The control
	is what separates "recovers the correspondence" from "matches a lot".

	The decoy is a fixed stride through the sorted file list rather than a random pick, so the whole
	check is deterministic and a failure is reproducible without a seed.

	MEASURED over 100 files: matchrate 0.980 true vs 0.028 decoy, prior +0.977 vs -0.630, and the true
	pairing wins on 100/100 files by matchrate. The per-file win count is the assertion, not the mean:
	a mean gap can be carried by a handful of files while most are ties.
	'''
	stride = 37			# coprime with 100, so no file is paired with itself
	true_rate, decoy_rate, wins, prior_wins = [], [], 0, 0
	for i, name in enumerate(names):
		src, tgt = pair_files(src_dir, tgt_dir, name, src_window)
		if src is None:
			continue
		other = names[(i + stride) % len(names)]
		if other == name:
			continue
		wrong_src, _ = pair_files(src_dir, tgt_dir, other, src_window)
		if wrong_src is None:
			continue
		good = align_pair(src, tgt)
		bad = align_pair(wrong_src, tgt)
		true_rate.append(good['matchrate'])
		decoy_rate.append(bad['matchrate'])
		wins += good['matchrate'] > bad['matchrate']
		prior_wins += good['prior'] > bad['prior']
	n = len(true_rate)
	ok = (n > 0 and wins >= n * 0.95 and prior_wins >= n * 0.9
		and statistics.mean(true_rate) > statistics.mean(decoy_rate) * 3)
	if not ok:
		print(f'  FAIL control: {wins}/{n} matchrate wins, {prior_wins}/{n} prior wins, '
			f'{statistics.mean(true_rate):.3f} vs {statistics.mean(decoy_rate):.3f}')
	print(f'{"ok  " if ok else "FAIL"} wrong-pair control: true matchrate '
		f'{statistics.mean(true_rate):.3f} vs decoy {statistics.mean(decoy_rate):.3f}, '
		f'true wins {wins}/{n} (matchrate) and {prior_wins}/{n} (prior)')
	return ok


def check_measure_recovery (rows):
	'''Matched pairs must land in the SAME BAR -- the hard, aligner-independent criterion.

	Threshold at p10 of the measured distribution rather than at the mean, so one pathological file
	cannot fail the suite while a real regression in the body of the distribution still does.
	MEASURED: mean 0.893, median 0.982, p10 0.721.
	'''
	mean, median, p10 = pct(rows, 'agree')
	ok = mean > 0.80 and median > 0.90 and p10 > 0.55
	if not ok:
		print(f'  FAIL measure recovery: mean {mean:.4f} median {median:.4f} p10 {p10:.4f}')
	print(f'{"ok  " if ok else "FAIL"} measure recovery: matched pairs in the same bar, '
		f'mean {mean:.4f}, median {median:.4f}, p10 {p10:.4f}')
	return ok


def check_tick_recovery (rows):
	"""Matched pairs must land in the same `(measure, tick)` GROUP -- the strictest ground truth here.

	This is the check the question "does align.py agree with @measure and @tick" actually turns on, and
	it is strictly harder than measure agreement: a group holds ~1-3 notes against a bar's ~16, so a
	wrong note that happens to be in the right bar is caught here and invisible there. MEASURED: 0.749
	mean and 0.809 median against measure agreement's 0.893/0.982, correlation 0.784 -- the gap between
	those two numbers is how much of the measure-level score was the bar's coarseness rather than
	alignment quality.

	Group membership, not a specific index, for the same reason align.py's ReuseCost note gives: a
	chord's notes share one key and may legitimately be paired in any order.

	Coverage is reported because the criterion is only as good as its reach -- MEASURED at 0.995 of
	matches, so this is not a check that quietly grades 10% of the data.

	PRECISION AND RECALL ARE BOTH REPORTED, and the distinction is not pedantic. Precision's denominator
	is the notes align.py MATCHED (0.749); recall's is every scorable target note, matched or not
	(0.590). The gap is the ~28% of notes it never paired at all, and quoting only precision flatters
	the result -- an aligner that matched 10% of the notes and got those right would score 1.0 on it.
	Recall is the end-to-end number. Both are gated, so neither can be traded for the other: raising
	matchrate by matching promiscuously would drop precision, and tightening precision by matching only
	the certain cases would drop recall.
	"""
	mean, median, p10 = pct(rows, 'tick_agree')
	rmean, rmedian, _rp10 = pct(rows, 'tick_recall')
	roll, _rollmed, _rollp10 = pct(rows, 'roll_agree')
	cover = statistics.mean(r['tick_cover'] for r in rows)
	# roll >= mean by construction (it is a superset), so this asserts the implementation, not a result.
	ok = mean > 0.60 and median > 0.65 and rmean > 0.48 and cover > 0.90 and roll >= mean
	if not ok:
		print(f'  FAIL tick recovery: precision {mean:.4f}/{median:.4f}, recall {rmean:.4f}, '
			f'cover {cover:.4f}')
	print(f'{"ok  " if ok else "FAIL"} @tick group recovery: precision {mean:.4f} (median '
		f'{median:.4f}, p10 {p10:.4f}) on {cover:.4f} of matches; RECALL {rmean:.4f} '
		f'(median {rmedian:.4f}) over every scorable target note')
	print(f'     ...allowing the mocker\'s rolled chords: precision {roll:.4f} '
		f'(+{roll - mean:.4f}) — that gap is the ground truth\'s own slack')
	return ok


def check_dp_reference_quality (rows):
	"""How right is the DP reference itself, judged against @tick? Its error is subtracted, not assumed.

	The DP pairing is a second aligner, so any disagreement between it and align.py is ambiguous about
	which of the two was wrong. @tick settles that: a DP pairing whose source note sits in a different
	(measure, tick) group than its target is the DP's error, full stop.

	MEASURED: 0.930 mean, 0.961 median. So ~7% of the reference's pairings are wrong, and roughly that
	much of align.py's apparent shortfall against it (0.705) is not align.py's. This is what turns the
	`exact_pair_recovery` number from an unattributable disagreement into a bounded one -- and it is
	why that check is loosely thresholded rather than treated as a verdict.

	It also guards the reference against silent rot: if a future change to `dp_pairs` made it worse,
	`exact_pair_recovery` would drift with no indication of which side moved. This check names the side.
	"""
	mean, median, p10 = pct(rows, 'dp_quality')
	ok = mean > 0.85 and median > 0.88
	if not ok:
		print(f'  FAIL DP reference quality: mean {mean:.4f} median {median:.4f} p10 {p10:.4f}')
	print(f'{"ok  " if ok else "FAIL"} DP reference vs @tick truth: mean {mean:.4f}, '
		f'median {median:.4f}, p10 {p10:.4f} (so ~{100 * (1 - mean):.0f}% of its pairings are its own error)')
	return ok


def check_exact_pair_recovery (rows):
	'''Agreement with the DP reference pairing, on the ~70% of notes it covers.

	Reported at a deliberately loose threshold, because a disagreement here is genuinely ambiguous
	about which of the two aligners was wrong -- the DP reference pairs by pitch within a bar, and a
	bar with a repeated pitch has more than one defensible answer. It is here to catch a COLLAPSE
	(exact recovery falling to chance), not to adjudicate individual notes.
	MEASURED: mean 0.705, median 0.784, p10 0.382.
	'''
	mean, median, p10 = pct(rows, 'exact')
	ok = mean > 0.55 and median > 0.60
	if not ok:
		print(f'  FAIL exact recovery: mean {mean:.4f} median {median:.4f} p10 {p10:.4f}')
	print(f'{"ok  " if ok else "FAIL"} exact pair recovery vs DP reference: mean {mean:.4f}, '
		f'median {median:.4f}, p10 {p10:.4f}')
	return ok


def check_monotonicity (rows):
	'''The pairing must be near-monotone: align.py's design says a backward offset is always an error.

	Tolerance of 4 indices, not 0, and that is the substance rather than slack: a chord's notes may be
	emitted in any order, so a match on a LOWER index is normal within one. What must not happen is
	the alignment walking BACK past a whole chord.
	MEASURED: mean 0.0027, median 0.0000 -- so a nonzero median would be a real change.
	'''
	mean, median, _p10 = pct(rows, 'backward')
	worst = max(r['backward'] for r in rows)
	ok = mean < 0.02 and median == 0.0 and worst < 0.15
	if not ok:
		print(f'  FAIL monotonicity: mean {mean:.4f} median {median:.4f} worst {worst:.4f}')
	print(f'{"ok  " if ok else "FAIL"} monotonicity: backward jumps > 4 indices on '
		f'{mean:.4f} of matches (median {median:.4f}, worst file {worst:.4f})')
	return ok


def check_tempo_ratio (rows):
	'''The EMA'd tgt/src tick ratio must land near 1 and never go negative or absurd.

	Near 1 because the mocker perturbs timing locally without rescaling the piece, so a ratio far from
	1 would mean the aligner is tracking a tempo relationship that is not there. The bound is loose on
	purpose: rubato is real and per-file ratios legitimately range roughly 0.78-1.41 on this corpus.
	'''
	ratios = [r['ratio'] for r in rows if r['ratio'] is not None]
	mean = statistics.mean(ratios)
	inside = sum(1 for x in ratios if 0.5 < x < 2.0)
	ok = 0.8 < mean < 1.25 and inside >= len(ratios) * 0.95 and all(x > 0 for x in ratios)
	if not ok:
		print(f'  FAIL ratio: mean {mean:.4f}, {inside}/{len(ratios)} in (0.5, 2.0)')
	print(f'{"ok  " if ok else "FAIL"} tempo ratio: mean {mean:.4f}, '
		f'{inside}/{len(ratios)} files inside (0.5, 2.0), none <= 0')
	return ok


def check_mask_contains_truth (rows):
	'''`tick_interval` must not exclude the tick the target note actually has. THE WEAK RESULT HERE.

	This is the one property whose failure mode is invisible in production, which is why it is measured
	rather than assumed: a mask can only ever DELETE the right answer, so an interval that misses the
	truth does not raise anything -- the run just scores slightly worse, with no symptom to chase.

	MEASURED: contains-true 0.595 mean over 100 files, 0.646 on the 91 that align well
	(measure-agree >= 0.7), correlation between the two 0.705. So the mask is not independently broken;
	it degrades exactly where the alignment under it has already failed, which is the expected
	direction — `tick_interval` is built from `predict_tick` and the residual EMA, and both are
	functions of the pairs `observe` found.

	The threshold is therefore set LOW and the number is printed, deliberately. Two limits on how much
	this condemns: part of the gap is DP-reference error rather than mask error -- now QUANTIFIED at 7.0%
	of the reference's own pairings, by checking it against @tick -- and `tick_interval` has no caller today
	-- grep finds it only in align.py and the two test files -- so this is a latent property rather
	than a live mis-kill. Raising it is a real improvement available in align.py; passing this check is
	not evidence that it does not need one.
	'''
	mean, median, p10 = pct(rows, 'mask')
	good = [r for r in rows if r['mask'] is not None and r['agree'] >= 0.7]
	good_mean = statistics.mean(r['mask'] for r in good) if good else 0.0
	fired = sum(r['mask_n'] for r in rows)
	ok = mean > 0.40 and median > 0.40 and good_mean > mean
	if not ok:
		print(f'  FAIL mask: mean {mean:.4f} median {median:.4f} well-aligned {good_mean:.4f}')
	print(f'{"ok  " if ok else "FAIL"} mask contains true tick: mean {mean:.4f}, median {median:.4f}, '
		f'p10 {p10:.4f}; {good_mean:.4f} on the {len(good)} well-aligned files ({fired} notes fired)')
	return ok


def check_forecast_ranks_truth (src_dir, tgt_dir, names, src_window):
	'''`forecast` must score the note's REAL onset below shifted decoys, on real rubato.

	align_check.py already shows forecast separates ticks on a synthetic source. What it cannot show is
	whether that separation points at the RIGHT tick once the source is a human-like performance and
	the target is quantised -- the two disagree by hundreds of ticks, and the forecast is anchored on a
	ratio estimated from the alignment itself.

	Decoys at +-480 and +-960 (a beat and two at this corpus's 480 ticks/beat) rather than tiny
	offsets: the mask's own half-width floors at 24 ticks, so a decoy inside that is not a
	distinguishable claim and counting it would just measure the floor.
	MEASURED: 0.775 of 9866 scored positions, with 908 positions honestly reporting no opinion.
	'''
	wins = total = silent = 0
	for name in names:
		src, tgt = pair_files(src_dir, tgt_dir, name, src_window)
		if src is None:
			continue
		state = AlignState(src, seed_offset=0.0)
		for j, e in enumerate(tgt):
			# Skip the cold start: below MinPairsForMask pairs there is no ratio and no residual, so
			# forecast has nothing to say and a "loss" there would be scoring the absence of evidence.
			if j >= 4:
				true_cost, _index, _pred = state.forecast(e['onset'])
				if true_cost is None:
					silent += 1
				else:
					decoys = [e['onset'] + d for d in (-960, -480, 480, 960) if e['onset'] + d > 0]
					costs = [state.forecast(x)[0] for x in decoys]
					costs = [c for c in costs if c is not None]
					if costs:
						total += 1
						wins += true_cost <= min(costs)
			state.observe(e['pitch'], e['onset'], e['softIndex'])
	rate = wins / total if total else 0.0
	ok = rate > 0.65
	if not ok:
		print(f'  FAIL forecast: {wins}/{total} = {rate:.4f}')
	print(f'{"ok  " if ok else "FAIL"} forecast ranks the true onset best: {wins}/{total} = '
		f'{rate:.4f} against +-480/+-960 decoys ({silent} positions with no opinion)')
	return ok


def check_clone_independence (src_dir, tgt_dir, names, src_window):
	'''Cloning mid-piece on REAL data must leave the parent untouched and reproduce it exactly.

	align_check.py checks this on a synthetic source. It is re-checked here because beam search clones
	per branch at every position, and a shared-mutable leak would show up as branches contaminating
	each other -- a bug that a synthetic source with distinct pitches can easily miss, since the
	corruption needs a repeated pitch and a real chord to become visible.
	'''
	ok = True
	for name in names[:8]:
		src, tgt = pair_files(src_dir, tgt_dir, name, src_window)
		if src is None:
			continue
		state = AlignState(src, seed_offset=0.0)
		half = len(tgt) // 2
		for e in tgt[:half]:
			state.observe(e['pitch'], e['onset'], e['softIndex'])
		snapshot = (state.cost, state.value, state.matched, state.misses, len(state.pairs),
			dict(state.used))
		branch = state.clone()
		for e in tgt[half:]:
			branch.observe(e['pitch'], e['onset'], e['softIndex'])
		after = (state.cost, state.value, state.matched, state.misses, len(state.pairs),
			dict(state.used))
		if snapshot != after:
			ok = False
			print(f'  FAIL clone independence: parent mutated on {name[:8]}')
			break
		second = state.clone()
		for e in tgt[half:]:
			second.observe(e['pitch'], e['onset'], e['softIndex'])
		if (abs(second.cost - branch.cost) > 1e-12 or second.matched != branch.matched
				or second.misses != branch.misses):
			ok = False
			print(f'  FAIL clone determinism: two clones diverged on {name[:8]}')
			break
	print(f'{"ok  " if ok else "FAIL"} clone independence on real pairs: parent unchanged by a '
		f'branch, and two clones from one parent agree')
	return ok


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--root', default=DEFAULT_ROOT)
	ap.add_argument('--samples', type=int, default=100,
		help='files, in sorted order. Below 100 is noisier than the thresholds assume.')
	ap.add_argument('--src-window', type=int, default=SRC_WINDOW,
		help='source crop in TOKENS, matching translateMidiseq2Beam --src-window')
	args = ap.parse_args()

	src_dir = resolve_dir(args.root, SRC_ARM)
	tgt_dir = resolve_dir(args.root, TGT_ARM)
	if not src_dir or not tgt_dir:
		print(f'FAIL no corpus: need {SRC_ARM} and {TGT_ARM} under {args.root}')
		return 1

	load_notes.tokenizer = Midiseq2Tokenizer()
	load_notes.keywords = keyword_tokens(load_notes.tokenizer)

	shared = sorted(set(os.listdir(src_dir)) & set(os.listdir(tgt_dir)))
	names = shared[:args.samples]
	if not names:
		print('FAIL no shared basenames between the two arms')
		return 1

	print(f'align corpus checks: {src_dir} -> {tgt_dir}')
	print(f'{len(names)} files, src_window {args.src_window} tokens')
	# Say it at runtime, not only in the docstring. The thresholds are calibrated on the FULL 100 and
	# the head of the sorted list is a weak subset -- `--samples 14` fails measure recovery at 0.752
	# against a 0.80 threshold purely by sampling. Without this line that failure reads as a
	# regression in align.py, which is the most expensive kind of false alarm: it sends someone
	# looking for a bug that is not there.
	if len(names) < CALIBRATED_SAMPLES:
		print(f'NOTE thresholds are calibrated on {CALIBRATED_SAMPLES} files; at {len(names)} a '
			f'failure may be sampling, not a regression (the sorted head is a weak subset)')
	print()

	rows = collect(src_dir, tgt_dir, names, args.src_window)
	if not rows:
		print('FAIL every pairing was too short to measure')
		return 1

	results = [
		check_ground_truth(src_dir, tgt_dir, names),
		check_streams_differ(rows),
		check_wrong_pair_control(src_dir, tgt_dir, names, args.src_window),
		check_tick_recovery(rows),
		check_measure_recovery(rows),
		check_dp_reference_quality(rows),
		check_exact_pair_recovery(rows),
		check_monotonicity(rows),
		check_tempo_ratio(rows),
		check_mask_contains_truth(rows),
		check_forecast_ranks_truth(src_dir, tgt_dir, names, args.src_window),
		check_clone_independence(src_dir, tgt_dir, names, args.src_window),
	]
	failed = results.count(False)
	print(f'\n{len(results) - failed}/{len(results)} checks passed '
		f'({len(rows)} of {len(names)} files measurable)')
	return 1 if failed else 0


if __name__ == '__main__':
	sys.exit(main())
