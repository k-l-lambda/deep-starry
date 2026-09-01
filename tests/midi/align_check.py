'''Checks for starry/midi/align.py — the translation alignment state and the elapse mask.

The decisive one is FEASIBILITY EXHAUSTIVENESS. `feasible_elapse_tokens` is a closed-form predicate
over the canonical-decomposition automaton, and it is the only part of this design that can be
verified rather than measured — so it is verified against brute force, at every automaton state, for
many intervals, with the ground truth built by literally enumerating deltas and decomposing them.

That matters more than it looks. A mask can only ever DELETE the right answer, never invent a wrong
one, so an off-by-one in this predicate does not raise anything: it silently removes the correct
token from the model's distribution and the run just scores slightly worse. There is no symptom to
chase. The exhaustive comparison is what makes that class of bug impossible rather than unlikely.

The rest are properties the callers depend on: that the automaton classes partition the shipped
vocab, that canonical_run agrees with the tokenizer's own renderer, that the grammar walk is
resumable (chunked == whole) and that its delta accumulation agrees with translateMidiseq2's
independent tick walk, that softIndex collapses chords and saturates rests, that the anchor vote
resists outliers, and that cost/prior stay inside the bounds a single lambda depends on.

Run:  python tests/midi/align_check.py [--root DIR] [--samples N]
'''

import argparse
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.midi.align import (BIG, STAGE_BIG, STAGE_MID, STAGE_LOW, Config, AlignState,
	GrammarState, CLS_ELAPSE, CLS_KEYWORD, CLS_FIELD, CLS_SPECIAL,
	anchor_from_votes, canonical_run, elapse_class, elapse_value, feasible_elapse_tokens,
	reachable_range, soft_delta, soft_indices, token_class, walk_grammar)
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer

sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'midi'))
from translateMidiseq2 import encode_lines, keyword_tokens, note_on_events


DEFAULT_ROOT = os.path.expanduser('~/data/midi/test202608')

# Two ceilings, and keeping them apart is what makes the comparison airtight rather than
# approximately right. BRUTE_MAX bounds the ENUMERATION; CASE_MAX bounds the intervals any case may
# ask about, one full BIG below it. With every interval capped at CASE_MAX, every delta the
# predicate can call feasible is <= CASE_MAX < BRUTE_MAX and is therefore present in the table — so
# the two sides can be compared with no allowance for truncation at all. Without the gap, a state
# sitting at the enumeration ceiling has all of its continuations outside the table and brute force
# reports "nothing feasible" for tokens that plainly are.
BRUTE_MAX = 0x3000
CASE_MAX = BRUTE_MAX - 0x1000


def stage_of_prefix (values):
	'''The automaton stage after emitting this prefix of elapse VALUES.'''
	if not values:
		return STAGE_BIG
	return elapse_class(values[-1])


def brute_force_states ():
	'''Every reachable (run_delta, stage) -> {delta: [next values]}, by enumerating decompositions.

	Ground truth by construction: walk every delta's canonical run, and for each prefix record the
	state it leaves the automaton in together with the token that follows. `None` as a "next value"
	records that the run may END at that prefix.
	'''
	table = {}
	for delta in range(0, BRUTE_MAX + 1):
		run = canonical_run(delta)
		for k in range(len(run) + 1):
			state = (sum(run[:k]), stage_of_prefix(run[:k]))
			slot = table.setdefault(state, {})
			slot.setdefault(delta, []).append(run[k] if k < len(run) else None)
	return table


def brute_force_answer (table, state, lo, hi):
	'''(allowed values, may_end) for `state` over the inclusive delta interval [lo, hi].'''
	allowed = set()
	may_end = False
	for delta, nexts in table.get(state, {}).items():
		if delta < lo or delta > hi:
			continue
		for value in nexts:
			if value is None:
				may_end = True
			else:
				allowed.add(value)
	return allowed, may_end


def check_elapse_partition ():
	'''The three automaton classes partition the vocab's E tokens exactly, with nothing left over.

	Asserted rather than commented because the closed form in feasible_elapse_tokens enumerates by
	class: a vocab token belonging to none of them would be silently unmaskable, and one belonging to
	two would be double-counted.
	'''
	tk = Midiseq2Tokenizer()
	e_toks = [t for t in tk.tokens if elapse_value(t) is not None]
	buckets = {STAGE_BIG: [], STAGE_MID: [], STAGE_LOW: [], None: []}
	for tok in e_toks:
		buckets[elapse_class(elapse_value(tok))].append(tok)
	ok = (len(buckets[None]) == 0
		and len(buckets[STAGE_BIG]) == 1
		and len(buckets[STAGE_MID]) == 255
		and len(buckets[STAGE_LOW]) == 15
		and len(e_toks) == 271)
	if not ok:
		print(f'  FAIL partition: {len(e_toks)} E tokens -> big {len(buckets[STAGE_BIG])} '
			f'mid {len(buckets[STAGE_MID])} low {len(buckets[STAGE_LOW])} '
			f'unclassified {len(buckets[None])} {buckets[None][:8]}')
	print(f'{"ok  " if ok else "FAIL"} elapse partition: {len(e_toks)} E tokens = '
		f'1 BIG + 255 MID + 15 LOW, none unclassified')
	return ok


def check_canonical_parity ():
	'''canonical_run must equal the tokenizer's own _elapse rendering, in value space.

	Two decompositions of the same delta is one too many: the mask reasons about the run the model
	will be scored against, so if this drifts from what the corpus was built with, the mask's notion
	of feasible stops being about the actual token stream.
	'''
	tk = Midiseq2Tokenizer()
	bad = []
	for delta in list(range(0, 0x1200)) + [0x1234, 0x2000, 0x2fff, 0x3000, 0x4abc]:
		mine = canonical_run(delta)
		theirs = [elapse_value(t) for t in tk._elapse(delta)]
		if mine != theirs:
			bad.append((delta, mine, theirs))
	ok = not bad
	for delta, mine, theirs in bad[:4]:
		print(f'  FAIL canonical {delta:#x}: {mine} vs tokenizer {theirs}')
	print(f'{"ok  " if ok else "FAIL"} canonical_run == Midiseq2Tokenizer._elapse '
		f'(0..0x1200 plus multi-BIG spot checks)')
	return ok


def check_reachable_range ():
	'''reachable_range must bound exactly the totals brute force can still reach from a state.'''
	table = brute_force_states()
	bad = []
	for (run_delta, stage), slot in table.items():
		if run_delta > BRUTE_MAX - 0x1000:
			continue		# near the ceiling the enumeration itself truncates what is reachable
		deltas = sorted(slot.keys())
		lo, hi = reachable_range(run_delta, stage)
		if deltas[0] != lo:
			bad.append((run_delta, stage, 'lo', deltas[0], lo))
		if hi is not None and deltas[-1] != hi:
			bad.append((run_delta, stage, 'hi', deltas[-1], hi))
	ok = not bad
	for run_delta, stage, which, got, want in bad[:6]:
		print(f'  FAIL reachable ({run_delta:#x}, stage {stage}) {which}: brute {got:#x} vs {want:#x}')
	print(f'{"ok  " if ok else "FAIL"} reachable_range exact over {len(table)} automaton states')
	return ok


def check_feasibility_exhaustive (seed=0, intervals=200):
	'''THE check. feasible_elapse_tokens == brute force, at every state, over many intervals.

	Intervals are drawn to include the shapes that break naive predicates: empty (hi < lo), single
	points, intervals entirely below the state's accumulated delta (nothing feasible, and may_end
	must be False), intervals straddling a 0x1000 boundary, and wide ones.

	Every interval is capped at CASE_MAX so no allowance for enumeration truncation is needed — see
	the note there. States BEYOND CASE_MAX are deliberately kept in the case list rather than
	skipped: with the interval below them nothing is feasible, which exercises the "already past hi"
	edge on both sides for free.
	'''
	tk = Midiseq2Tokenizer()
	e_values = {t: elapse_value(t) for t in tk.tokens if elapse_value(t) is not None}
	table = brute_force_states()
	states = sorted(table.keys())
	rng = random.Random(seed)

	def case (state, lo, hi):
		cases.append((state, min(lo, CASE_MAX), min(hi, CASE_MAX)))

	cases = []
	# deterministic edge shapes, applied to every state
	for state in states:
		d = state[0]
		case(state, d, d)				# exactly the accumulated delta: may_end only
		case(state, d + 1, d)				# empty interval
		case(state, 0, max(0, d - 1))			# entirely below: nothing reachable
		case(state, d, d + 0xf)				# within one nibble
		case(state, d, d + BIG)				# straddles a BIG boundary
	# random intervals over random states
	for _ in range(intervals):
		state = rng.choice(states)
		lo = rng.randint(0, CASE_MAX)
		case(state, lo, lo + rng.choice([0, 1, 0xf, 0x50, 0x400, 0x1000, 0x2000]))

	bad = []
	for state, lo, hi in cases:
		run_delta, stage = state
		got_allowed, got_end = feasible_elapse_tokens(run_delta, stage, lo, hi, e_values)
		got_values = {e_values[t] for t in got_allowed}
		want_values, want_end = brute_force_answer(table, state, lo, hi)
		if got_values != want_values or got_end != want_end:
			bad.append((state, lo, hi, sorted(got_values ^ want_values), got_end, want_end))
	ok = not bad
	for state, lo, hi, diff, ge, we in bad[:6]:
		print(f'  FAIL feasible ({state[0]:#x}, stage {state[1]}) [{lo:#x},{hi:#x}]: '
			f'symmetric diff {[hex(v) for v in diff[:6]]} may_end {ge} vs {we}')
	print(f'{"ok  " if ok else "FAIL"} feasibility exhaustive: {len(cases)} cases over '
		f'{len(states)} states == brute force (intervals <= {CASE_MAX:#x}, '
		f'enumerated to {BRUTE_MAX:#x})')
	return ok


def check_feasibility_never_invents ():
	'''Every token the mask allows must actually appear at that position in SOME canonical run.

	The safety property the whole design rests on: a mask that allowed a token no decomposition can
	produce would be inventing a token, and then beam=1 could diverge from greedy on a legal stream.
	'''
	tk = Midiseq2Tokenizer()
	e_values = {t: elapse_value(t) for t in tk.tokens if elapse_value(t) is not None}
	table = brute_force_states()
	bad = []
	for state in sorted(table.keys()):
		run_delta, stage = state
		allowed, _ = feasible_elapse_tokens(run_delta, stage, 0, CASE_MAX, e_values)
		reachable, _ = brute_force_answer(table, state, 0, CASE_MAX)
		invented = {e_values[t] for t in allowed} - reachable
		if invented:
			bad.append((state, sorted(invented)[:6]))
	ok = not bad
	for state, invented in bad[:6]:
		print(f'  FAIL invented at ({state[0]:#x}, stage {state[1]}): {[hex(v) for v in invented]}')
	print(f'{"ok  " if ok else "FAIL"} mask never invents a token off the automaton')
	return ok


def check_empty_interval_falls_back ():
	'''An empty or unsatisfiable interval must return an EMPTY set, not a full one.

	The caller is required to fall back to unmasked when the survivor set is empty, so the predicate
	has to report emptiness honestly. Returning everything "to be safe" would hide the condition and
	the fallback counter would never fire.
	'''
	tk = Midiseq2Tokenizer()
	e_values = {t: elapse_value(t) for t in tk.tokens if elapse_value(t) is not None}
	a, end_a = feasible_elapse_tokens(0x100, STAGE_BIG, 0x200, 0x100, e_values)	# hi < lo
	b, end_b = feasible_elapse_tokens(0x500, STAGE_BIG, 0, 0x100, e_values)		# already past hi
	c, end_c = feasible_elapse_tokens(0x40, STAGE_LOW, 0x40, 0x40, e_values)		# closed run, in range
	d, end_d = feasible_elapse_tokens(0x40, STAGE_LOW, 0x50, 0x60, e_values)		# closed run, out of range
	ok = (not a and not end_a) and (not b and not end_b) and (not c and end_c) and (not d and not end_d)
	if not ok:
		print(f'  FAIL empty-interval: |a|={len(a)}/{end_a} |b|={len(b)}/{end_b} '
			f'|c|={len(c)}/{end_c} |d|={len(d)}/{end_d}')
	print(f'{"ok  " if ok else "FAIL"} empty/unsatisfiable intervals report empty (caller falls back)')
	return ok


def check_elapse_after_keyword ():
	'''An elapse token may not directly follow a keyword, but may follow a field or a boundary.

	The one structural rule GrammarState enforces. `note_on E10` is malformed — the event owes at
	least one argument — while `note_on #3b` may legitimately end there and start a delta run, which
	is why the rule is about the PREVIOUS token's class and not about counting arguments.
	'''
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	cases = [
		([], True, 'boundary'),
		(['note_on'], False, 'after keyword'),
		(['note_on', '#3b'], True, 'after field'),
		(['note_on', '#3b', '$22'], True, 'after second field'),
		(['E010'], True, 'inside run'),
		(['<eom>'], True, 'after <eom>'),
		(['<sep>'], True, 'after <sep>'),
	]
	ok = True
	for toks, want, label in cases:
		state = walk_grammar(toks, kw)
		if state.elapse_allowed != want:
			ok = False
			print(f'  FAIL elapse_allowed {label}: got {state.elapse_allowed}, want {want}')
	print(f'{"ok  " if ok else "FAIL"} elapse legality by previous token class '
		f'(banned only after a keyword)')
	return ok


def check_grammar_state_classes ():
	'''token_class must agree with the vocab's own structure on every token.'''
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	counts = {}
	for tok in tk.tokens:
		counts[token_class(tok, kw)] = counts.get(token_class(tok, kw), 0) + 1
	ok = (counts.get(CLS_ELAPSE) == 271 and counts.get(CLS_KEYWORD) == 16
		and counts.get(CLS_SPECIAL) == 8)
	if not ok:
		print(f'  FAIL classes: {counts}')
	print(f'{"ok  " if ok else "FAIL"} token_class over vocab: {counts.get(CLS_ELAPSE)} elapse, '
		f'{counts.get(CLS_KEYWORD)} keyword, {counts.get(CLS_SPECIAL)} special, '
		f'{counts.get(CLS_FIELD)} field')
	return ok


def check_grammar_walk_incremental (root, samples):
	'''Walking a real stream in CHUNKS must equal walking it whole, and its delta accumulation must
	agree with translateMidiseq2's independent tick walk.

	Both halves matter. The first is what lets beam generation commit one token at a time and still
	hold a correct mask state -- the same resumability note_on_events already guarantees. The second
	is the counter-agreement check: two accumulators over one stream only have to disagree once for
	the mask to be reasoning about a different tick than the notes are at, so they are compared
	rather than assumed. GrammarState banks a run's delta when the run CLOSES, so the comparable
	quantity is total_delta + run_delta (an unbanked open run at end of stream).
	'''
	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	d = os.path.join(root, 'nota1m-100', 'midi-seq2-irregular')
	if not os.path.isdir(d):
		d = os.path.join(root, 'midi-seq2-irregular')
	if not os.path.isdir(d):
		print('skip grammar walk: no corpus arm found')
		return True
	ok = True
	for name in sorted(os.listdir(d))[:samples]:
		lines = open(os.path.join(d, name)).read().splitlines()[:400]
		ids = encode_lines(lines, tk)
		toks = [tk.tokens[i] for i in ids]
		whole = walk_grammar(toks, kw)
		_events, tick, _state = note_on_events(ids, tk, kw)
		if whole.total_delta + whole.run_delta != tick:
			ok = False
			print(f'  FAIL {name[:8]} delta accumulation: grammar '
				f'{whole.total_delta + whole.run_delta} vs note walk {tick}')
			continue
		for chunk in (1, 2, 3, 17, 256):
			state, walked = None, 0
			while walked < len(toks):
				part = toks[walked:walked + chunk]
				state = walk_grammar(part, kw, state)
				walked += len(part)
			same = (state.prev_class == whole.prev_class and state.run_delta == whole.run_delta
				and state.stage == whole.stage and state.open_keyword == whole.open_keyword
				and state.total_delta == whole.total_delta)
			if not same:
				ok = False
				print(f'  FAIL {name[:8]} chunk={chunk}: state {state.prev_class}/'
					f'{state.run_delta:#x}/{state.stage} vs {whole.prev_class}/'
					f'{whole.run_delta:#x}/{whole.stage}')
				break
	print(f'{"ok  " if ok else "FAIL"} grammar walk: chunked == whole at 1/2/3/17/256, and '
		f'delta accumulation == note_on_events tick')
	return ok


def check_soft_index ():
	'''softIndex must collapse a chord onto one position and saturate a long rest.

	Both saturations are load-bearing. Without the low end a 4-note source chord would be four
	positions the target has to match one at a time; without the high end a fermata would push the
	coordinate away without bound and every offset after it would be measured against a runaway
	baseline.
	'''
	sim = Config['SIMULTANEOUS_TICKS']
	chord = soft_indices([1000, 1000, 1001, 1002])
	spread = chord[-1] - chord[0]
	rest = soft_indices([0, int(sim * 40)])
	step = soft_indices([0, int(sim)])[1]
	ok = (spread < 0.03				# a chord is ~one position
		and rest[1] > 0.99			# a long rest saturates at 1
		and 0.6 < step < 0.9			# one SIMULTANEOUS_TICKS is tanh(1) ~ 0.762
		and soft_delta(-50) == 0.0)		# negative interval clamps rather than going backwards
	if not ok:
		print(f'  FAIL soft index: chord spread {spread:.4f}, long rest {rest[1]:.4f}, '
			f'unit step {step:.4f}, negative {soft_delta(-50)}')
	print(f'{"ok  " if ok else "FAIL"} softIndex: chord spread {spread:.4f} < 0.03, '
		f'long rest {rest[1]:.4f} -> 1, unit step {step:.3f}')
	return ok


def check_anchor_vote ():
	'''The anchor must be the histogram MODE, robust to outliers, with a width that tracks spread.

	This is why the anchor is a vote rather than "the offset of the best recent match": a couple of
	stray pairings cannot move the mode, but they can trivially BE the single best match. The support
	width is what the mask interval is derived from, so a flat histogram has to yield a wide one.
	'''
	tight = {0.50: 3.0, 0.52: 4.0, 0.51: 3.0, 2.90: 1.0}		# one far outlier
	mode, conf, lo, hi = anchor_from_votes(tight)
	flat = {0.1: 1.0, 0.6: 1.0, 1.1: 1.0, 1.6: 1.0, 2.1: 1.0}
	_m2, conf2, lo2, hi2 = anchor_from_votes(flat)
	ok = (abs(mode - 0.51) < 0.02			# mode sits in the cluster, not dragged by the outlier
		and hi - lo < 0.9			# tight histogram -> narrow support
		and hi2 - lo2 > hi - lo			# flat histogram -> wider support
		and conf > conf2)			# and lower confidence
	if not ok:
		print(f'  FAIL anchor: mode {mode:.3f} conf {conf:.3f} span [{lo:.2f},{hi:.2f}]; '
			f'flat conf {conf2:.3f} span [{lo2:.2f},{hi2:.2f}]')
	print(f'{"ok  " if ok else "FAIL"} anchor vote: mode {mode:.3f} (outlier at 2.90 ignored), '
		f'span {hi - lo:.2f} vs flat {hi2 - lo2:.2f}, conf {conf:.2f} > {conf2:.2f}')
	return ok


def make_source (n=40, step=240, pitches=None):
	'''A synthetic source window: evenly spaced onsets with softIndex filled in.'''
	pitches = pitches or [60 + (i * 5) % 24 for i in range(n)]
	onsets = [i * step for i in range(n)]
	sis = soft_indices(onsets)
	return [dict(onset=o, pitch=p, softIndex=s) for o, p, s in zip(onsets, pitches, sis)]


def check_cost_bounds ():
	'''cost must stay under the geometric bound and prior inside (-1, 1), consistent < inconsistent.

	The bound is what makes a single lambda portable: each step adds at most two tanh terms and the
	previous cost is attenuated by 0.6, so the sum cannot exceed 2/(1-0.6) = 5. An unbounded
	alignment term would need lambda re-fitted per file length, which is precisely the failure the
	length-normalised log-probability pairing is meant to avoid.
	'''
	src = make_source()
	# a consistent target: same pitches, same spacing, constant offset
	good = AlignState(src)
	good_sis = soft_indices([x['onset'] for x in src])
	for i, e in enumerate(src[:24]):
		good.observe(e['pitch'], e['onset'], good_sis[i])
	# an inconsistent target: pitches in scrambled order, so the offset jumps every note
	bad = AlignState(src)
	order = [0, 17, 3, 29, 8, 22, 1, 35, 12, 5, 31, 9, 20, 2, 27, 14]
	sis = soft_indices([x['onset'] for x in src])
	for i in order:
		bad.observe(src[i]['pitch'], src[i]['onset'], sis[i])
	bound = 2.0 / (1.0 - Config['CostStepAttenuation'])
	ok = (good.cost < bound and bad.cost < bound
		and -1.0 < good.prior < 1.0 and -1.0 < bad.prior < 1.0
		and good.cost < bad.cost and good.prior > bad.prior)
	if not ok:
		print(f'  FAIL bounds: good cost {good.cost:.3f} prior {good.prior:.3f}; '
			f'bad cost {bad.cost:.3f} prior {bad.prior:.3f}; bound {bound:.2f}')
	print(f'{"ok  " if ok else "FAIL"} cost bounded by {bound:.1f} and prior in (-1,1): '
		f'consistent {good.cost:.3f}/{good.prior:+.3f} vs scrambled {bad.cost:.3f}/{bad.prior:+.3f}')
	return ok


def check_clone_independence ():
	'''A cloned AlignState must not share mutable history with its parent.

	Beam search clones per branch, so a shared `pairs` list would make every beam's anchor the union
	of all beams' matches -- an alignment that no single hypothesis actually has.
	'''
	src = make_source()
	sis = soft_indices([x['onset'] for x in src])
	base = AlignState(src)
	for i in range(6):
		base.observe(src[i]['pitch'], src[i]['onset'], sis[i])
	a = base.clone()
	b = base.clone()
	for i in range(6, 12):
		a.observe(src[i]['pitch'], src[i]['onset'], sis[i])
	# Discriminate on `value`, not `cost`. A perfectly consistent alignment has cost EXACTLY 0 at any
	# length -- 0 * 0.6 + tanh(0) + tanh(0) -- so cost cannot witness divergence here, while value
	# accumulates +1 per matched note and can.
	ok = (len(b.pairs) == len(base.pairs) == 6 and len(a.pairs) == 12
		and b.value == base.value and a.value > b.value
		and b.last_offset == base.last_offset
		and a.src_events is base.src_events)		# source is intentionally shared (read-only)
	if not ok:
		print(f'  FAIL clone: base {len(base.pairs)} a {len(a.pairs)} b {len(b.pairs)} pairs, '
			f'values {base.value:.3f}/{a.value:.3f}/{b.value:.3f}')
	print(f'{"ok  " if ok else "FAIL"} clone independence: histories diverge '
		f'(value {b.value:.1f} vs {a.value:.1f}), source shared')
	return ok


def check_interval_evidence_gate ():
	'''No mask interval until there is evidence for one, and its width must track the residual.

	The gate is the honest answer to "how wide should this be" when nothing has been measured yet:
	a mask derived from default constants would mis-kill invisibly. And the width being driven by the
	model's OWN recent prediction error is what keeps the mask from being rigid -- while the model
	tracks the source the interval tightens around it, and when the model starts drifting the
	interval opens instead of fighting it.
	'''
	src = make_source(n=60)
	sis = soft_indices([x['onset'] for x in src])

	fresh = AlignState(src)
	lo0, hi0 = fresh.tick_interval(src[0]['onset'])
	after_two = AlignState(src)
	for i in range(2):
		after_two.observe(src[i]['pitch'], src[i]['onset'], sis[i])
	lo1, hi1 = after_two.tick_interval(src[2]['onset'])

	# a target tracking the source exactly: residual stays tiny -> narrow interval
	exact = AlignState(src)
	for i in range(30):
		exact.observe(src[i]['pitch'], src[i]['onset'], sis[i])
	lo_e, hi_e = exact.tick_interval(src[30]['onset'])

	# a target whose ticks wobble: residual grows -> wider interval
	noisy = AlignState(src)
	rng = random.Random(7)
	for i in range(30):
		noisy.observe(src[i]['pitch'], src[i]['onset'] + rng.randint(-260, 260), sis[i])
	lo_n, hi_n = noisy.tick_interval(src[30]['onset'])

	floor = Config['RatioResidualFloor']
	ok = (lo0 is None and lo1 is None			# gated while evidence is thin
		and lo_e is not None and lo_n is not None
		and (hi_e - lo_e) >= 2 * floor			# never collapses below the floor
		and (hi_n - lo_n) > (hi_e - lo_e))		# noisy model -> wider mask
	if not ok:
		print(f'  FAIL interval gate: fresh {lo0}, 2 pairs {lo1}, '
			f'exact width {None if lo_e is None else hi_e - lo_e}, '
			f'noisy width {None if lo_n is None else hi_n - lo_n}')
	print(f'{"ok  " if ok else "FAIL"} interval gated below {Config["MinPairsForMask"]} pairs; '
		f'width {hi_e - lo_e:.0f} (tracking) < {hi_n - lo_n:.0f} (wobbling)')
	return ok


def check_miss_is_not_fatal ():
	'''A generated note absent from the source must cost, but not destroy the alignment.

	The score arm can legitimately carry a note the irregular arm does not have near that position,
	so a miss has to be survivable: charging it as unbounded would let one such note eliminate an
	otherwise correct beam.
	'''
	src = make_source()
	sis = soft_indices([x['onset'] for x in src])
	st = AlignState(src)
	for i in range(10):
		st.observe(src[i]['pitch'], src[i]['onset'], sis[i])
	before_prior, before_value = st.prior, st.value
	st.observe(7, src[10]['onset'], sis[10])		# pitch 7 is in no source window
	bound = 2.0 / (1.0 - Config['CostStepAttenuation']) + Config['MissCost']
	ok = (st.misses == 1 and st.value == before_value	# a miss earns no evidence
		and st.prior < before_prior			# but does cost
		and st.cost < bound)
	if not ok:
		print(f'  FAIL miss: misses {st.misses} value {st.value:.3f}/{before_value:.3f} '
			f'prior {st.prior:.3f}/{before_prior:.3f} cost {st.cost:.3f}')
	print(f'{"ok  " if ok else "FAIL"} unmatched note: costs ({before_prior:+.3f} -> '
		f'{st.prior:+.3f}) without unbounded penalty')
	return ok


def check_forecast_discriminates_ticks ():
	'''The tick forecast must separate elapse candidates, and observe-based scoring must not.

	This pins the reason `forecast` is built on predict_tick rather than on observe: soft_delta
	saturates, so in softIndex space every elapse past ~350 ticks is the same place, and an
	observe-based estimate is FLAT across exactly the choices the search needs separated. The
	measured case is a source note ~530 ticks ahead of the current target tick, where the four elapse
	options bracket it.
	'''
	# Spaced widely on purpose. The forecast is a MIN over the notes ahead, so it is not a function
	# of the distance to any one of them -- a candidate that happens to land on a later onset will
	# (correctly) aim there instead. Monotonicity in |tick error| is therefore a claim about
	# candidates competing for the SAME note, which needs the neighbouring onsets kept out of range.
	# 960 ticks apart mirrors the case that motivated this (source onsets 2065 -> 2769).
	src = make_source(n=40, step=960)
	sis = [x['softIndex'] for x in src]
	st = AlignState(src)
	# track the source exactly for a while, so a ratio and a residual exist
	for i in range(12):
		st.observe(src[i]['pitch'], src[i]['onset'], sis[i])
	# the first UNMATCHED source note is #12: observing 0..11 leaves the walk starting at 12
	target_src = src[12]['onset']
	here = src[11]['onset']
	options = [0, 240, 480, 720, 840, 960]

	costs = [st.forecast(here + dt)[0] for dt in options]
	no_opinion = [c is None for c in costs]
	errs = [abs((here + dt) - st.predict_tick(target_src)) for dt in options]
	# the forecast must be monotone in |tick error|: rank by cost == rank by error
	by_cost = sorted(range(len(options)), key=lambda i: costs[i])
	by_err = sorted(range(len(options)), key=lambda i: errs[i])
	monotone = not any(no_opinion) and by_cost == by_err

	# and the observe-based estimate must be shown FLAT over the saturated range, which is the whole
	# reason tick space is used -- if this ever stops being flat the forecast could be simplified
	soft_costs = []
	for dt in options:
		if dt < 350:
			continue
		probe = st.clone()
		probe.observe(src[13]['pitch'], here + dt, sis[12] + soft_delta(dt))
		soft_costs.append(probe.cost)
	flat = len(soft_costs) >= 2 and (max(soft_costs) - min(soft_costs)) < 1e-6

	ok = monotone and flat
	print(f'{"ok  " if ok else "FAIL"} forecast is monotone in |tick error| where observe is flat: '
		f'best dt {options[by_cost[0]]} (err {errs[by_cost[0]]:.0f}), '
		f'worst dt {options[by_cost[-1]]} (err {errs[by_cost[-1]]:.0f}); '
		f'observe spread over the saturated range {max(soft_costs) - min(soft_costs):.2e}')
	if not ok:
		print(f'  costs {[None if c is None else round(c, 4) for c in costs]}')
		print(f'  errs  {[round(e) for e in errs]}')
	return ok


def check_forecast_charges_skipping ():
	'''A far tick must not win by aiming past the source notes it skipped.

	The forecast is optimistic across the lookahead on purpose -- a tick cannot be charged for a
	pitch mistake not yet made -- but "optimistic" must not extend to ignoring the notes in between.
	Without the skip term a candidate a whole phrase ahead can land near a LATER onset and outrank
	the one that fits the next note.
	'''
	src = make_source(n=40, step=240)
	sis = [x['softIndex'] for x in src]
	st = AlignState(src)
	for i in range(12):
		st.observe(src[i]['pitch'], src[i]['onset'], sis[i])
	near = st.forecast(src[12]['onset'])		# right at the next unmatched onset
	far = st.forecast(src[17]['onset'])			# right at one five notes later
	# The claim is NOT that the skip charge drags a far tick back onto the next note -- a tick a
	# phrase ahead genuinely does correspond to a later onset, and `observe` reads it the same way.
	# It is that landing on a later onset must cost MORE than landing on the next one, so a candidate
	# cannot buy a good forecast by skipping the notes it owes.
	ok = near[0] is not None and far[0] is not None and near[0] < far[0] and near[1] == 12
	print(f'{"ok  " if ok else "FAIL"} skipping is charged: next-onset cost {near[0]:.4f} '
		f'(aims #{near[1]}) < five-ahead cost {far[0]:.4f} (aims #{far[1]})')
	return ok


def check_forecast_has_no_opinion_early ():
	'''With no ratio, no residual, or nothing unmatched ahead, the forecast must abstain.

	The caller falls back to the language model on None. Returning a number here instead would be a
	ranking derived from unmeasured defaults, and its mis-rankings would be invisible -- the same
	argument that gates the mask interval.
	'''
	src = make_source(n=12, step=240)
	sis = [x['softIndex'] for x in src]
	fresh = AlignState(src)
	c_fresh = fresh.forecast(0)[0]

	one = AlignState(src)
	one.observe(src[0]['pitch'], src[0]['onset'], sis[0])
	c_one = one.forecast(240)[0]			# a pair, but no baseline yet -> no ratio

	exhausted = AlignState(src)
	for i in range(len(src)):
		exhausted.observe(src[i]['pitch'], src[i]['onset'], sis[i])
	c_done = exhausted.forecast(src[-1]['onset'] + 240)[0]

	ok = c_fresh is None and c_one is None and c_done is None
	print(f'{"ok  " if ok else "FAIL"} forecast abstains without evidence: fresh {c_fresh}, '
		f'one pair {c_one}, source exhausted {c_done}')
	return ok


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--root', default=DEFAULT_ROOT)
	ap.add_argument('--samples', type=int, default=4)
	args = ap.parse_args()

	print(f'align checks (corpus {args.root})\n')
	results = [
		check_elapse_partition(),
		check_canonical_parity(),
		check_reachable_range(),
		check_feasibility_exhaustive(),
		check_feasibility_never_invents(),
		check_empty_interval_falls_back(),
		check_grammar_state_classes(),
		check_elapse_after_keyword(),
		check_grammar_walk_incremental(args.root, args.samples),
		check_soft_index(),
		check_anchor_vote(),
		check_cost_bounds(),
		check_clone_independence(),
		check_interval_evidence_gate(),
		check_miss_is_not_fatal(),
		check_forecast_discriminates_ticks(),
		check_forecast_charges_skipping(),
		check_forecast_has_no_opinion_early(),
	]
	failed = results.count(False)
	print(f'\n{len(results) - failed}/{len(results)} checks passed')
	return 1 if failed else 0


if __name__ == '__main__':
	sys.exit(main())
