'''Source/target alignment state for midiseq2 translation, and the elapse-token feasibility mask.

Ported from the ideas in music-widgets `inc/Matcher` (real-time score following) and
`intelli-piano/inc/melody/heuristicNavigator` (its neural successor), retargeted at
`tools/midi/translateMidiseq2.py`'s sliding-window generation.

Why those algorithms transfer. Both problems maintain a correspondence between two note streams
from local evidence only, and both live or die on one invariant: the OFFSET between the two
streams' positions should stay locally constant. Matcher builds its whole cost function on that
single quantity, which is why none of it depends on knowing a tempo, a meter, or a bar number.

What is different here, and it is a simplification: translation is monotone and near-bijective on
note_on. There are no repeats, restarts or page jumps, so Matcher's relocation machinery is
deliberately NOT ported — Follower3 had already reached the same conclusion by construction
(`relocationThreshold: Infinity`). Offset going backwards is always an error here, never a
hypothesis worth keeping.

Two mechanisms live in this module, with separate failure modes and separate instrumentation:

  MASK (hard, exact)     `feasible_elapse_tokens` — which E... tokens can still reach a target
                         delta interval, given the canonical-decomposition prefix already emitted.
                         Pure combinatorics over the elapse automaton; exhaustively verifiable
                         against brute force, and `tests/midi/align_check.py` does verify it.

  RANK (soft, heuristic) `AlignState.prior` — Matcher's bounded evidence-minus-inconsistency score,
                         for ordering beams. Tunable, and NOT verifiable the same way.

Keeping them apart is the point. Blended into one number, a wrong result can no longer be
attributed to the automaton versus the heuristic.

The mask NEVER invents a token: it can only remove ones the model already scored. So its only
failure direction is deleting the right answer, which is why every caller must count
`killed_argmax` and must fall back to unmasked when the survivor set is empty. See
AlignState.delta_interval for why the interval's WIDTH is derived from the model's own recent
prediction error rather than from a constant — a fixed tolerance is what would make this rigid.

CONSTANTS ARE PROVISIONAL. The ones inherited from Matcher were tuned for real-time piano
following, where the error distribution is a human rushing or dragging. Here the error
distribution is a model miscounting bars. Every value under `Config` is a starting point to be
swept, not a measured one; `SIMULTANEOUS_TICKS` in particular is a ms->tick retranslation that has
to be re-measured on the corpus before any of this is trustworthy.
'''

import math


# --- config -------------------------------------------------------------------------------

Config = dict(
	# softIndex: the tick separation at which two onsets stop being "simultaneous". Matcher used
	# 0.24 * 800ms of wall clock; at 480 ticks/beat and ~120bpm a beat is ~480 ticks, so 0.24 beat
	# is ~115 ticks. PROVISIONAL — must be re-measured against the corpus IOI distribution, since
	# too small degenerates softIndex into linear ticks (losing chord collapse) and too large
	# saturates everything (losing resolution).
	SIMULTANEOUS_TICKS = 115.0,

	# Matcher's cost recursion. The 0.6 attenuation is a deliberate forgetting factor: it makes the
	# score a LOCAL consistency measure, so a beam is ranked on whether the last few bars cohere
	# rather than on the whole file's history. tanh bounds each term, so one step costs < 2 and the
	# geometric sum is < 2/(1-0.6) = 5 — which is what lets a single lambda weight this against a
	# length-normalised log-probability across different pieces.
	CostStepAttenuation = 0.6,
	SkipCost = 0.5,
	SelfCostScale = 0.5,

	# Asymmetric offset penalty. SIGN CONVENTION HERE: offset = src_softIndex - tgt_softIndex, and
	# bias = offset_now - offset_prev. bias > 0 means the source advanced more than the target, i.e.
	# the target UNDER-produced (dropped a note or under-timed a delta). bias < 0 means it
	# OVER-produced. Matcher's 1.0/1.6 split encoded "humans drag more often than they rush"; the
	# equivalent asymmetry for a translating model is unmeasured and may well point the other way,
	# so these two are the first pair to sweep.
	UnderCost = 1.0,
	OverCost = 1.6,

	# Cost charged when a generated note has no plausible source counterpart at all. Fixed rather
	# than derived: there is no offset to measure when nothing matched.
	MissCost = 1.0,

	# Charged when a match RE-USES a source note this lineage has already paired. Nothing charged for
	# that before: `skip` charges only for jumping too far AHEAD, so MEASURED on a fresh state with
	# three notes folded in, re-matching the same source note, advancing to the next, and going two
	# back all returned self_cost 0.0, skip 0, cost 0.0 -- three different musical claims priced
	# identically.
	#
	# RE-USE, not index order, is the right quantity, and that distinction is the whole design here.
	# A chord's notes may legitimately be emitted in any order, so a match that lands on a LOWER index
	# than the last one is perfectly normal: MEASURED on the score-only winning path, 7 of its 8
	# non-advancing steps were within a chord (same generated onset, distinct source notes) and only 1
	# was a genuine backward step. An index-order charge punishes the chords, which is why the first
	# attempt at this made the run worse rather than better (at 0.25 the path sat on src 3 for 45 of
	# its 51 notes).
	#
	# It only became load-bearing once the PITCH branch was adjudicated. While the pitch was ranked by
	# the language model, re-use was penalised only indirectly and the model's own distribution carried
	# the note forward. Ranked on this cost, re-use is strictly the cheapest thing available and the
	# search finds it -- and it is cheapest for a specific reason worth writing down: advancing to the
	# right note CHANGES THE OFFSET, which `self_cost = (bias * coeff)**2` charges, while standing on
	# the same note keeps the offset constant at self_cost 0.0. So this charge has to be big enough to
	# dominate the drift term, not merely nonzero. MEASURED on the first pitch-adjudicated run: 4
	# distinct source notes matched 51 times (src 1 alone 29), one endless chord, 0 measures, against
	# 19 distinct of 22 on the score-only control.
	#
	# Charged per PRIOR pairing, so a second use is cheaper than a fifth, and bounded by tanh like
	# every other term in the recursion.
	#
	# DEFAULT 0.0 = exactly the old behaviour, because how hard to charge is a ranking-policy choice
	# and not something to change silently. `translateMidiseq2Beam.py --reuse-cost` sets it.
	ReuseCost = 0.0,

	# prior = tanh(value * PriorValue) - tanh(cost * PriorCost). Matcher's form: saturating evidence
	# gain minus saturating inconsistency, so neither term can dominate without bound.
	PriorValue = 0.12,
	PriorCost = 0.5,

	# Anchor vote: pairs within this softIndex span of the newest one get a ballot, and the
	# histogram is gaussian-smoothed at this sigma before the mode is taken.
	AnchorSoftSpan = 0.8,
	AnchorSigma = 0.5,
	# The mask interval spans the offsets whose smoothed density clears this fraction of the peak.
	# Larger = tighter mask. This is the knob that trades mask strength against mis-kill risk.
	AnchorSupportFrac = 0.25,

	# Tempo ratio (tgt ticks per src tick), Follower1.updateSpeed's scheme: only take a slope over a
	# baseline at least this far apart in softIndex (short baselines are noise), then EMA it.
	RatioBaselineSI = 4.0,
	RatioEmaAlpha = 0.4,
	# Mask half-width = RatioResidualK * (EMA of |prediction residual|), floored so a lucky run of
	# exact predictions cannot collapse the interval to nothing.
	RatioResidualK = 3.0,
	RatioResidualFloor = 24.0,
	# No mask until this many pairs have been observed: with no history there is no anchor, and a
	# mask derived from nothing would be a mask derived from the defaults.
	MinPairsForMask = 4,

	# How many unmatched source notes ahead `forecast` offers as candidate pitches for a tick whose
	# pitch is not yet chosen. It bounds an OPTIMISTIC estimate, so larger is not automatically
	# better: past the local chord it starts offering notes a correct continuation would not reach
	# yet, and every one of them can only lower the forecast. 6 covers a four-note chord plus the two
	# following onsets on the test material; unswept.
	ForecastLookahead = 6,
)


# --- elapse automaton ---------------------------------------------------------------------
#
# The canonical decomposition (Midiseq2Tokenizer._elapse, mirroring the TS grammar) is a strict
# automaton, not a free choice of tokens:
#
#     while ticks >= 0x1000: emit E1000; ticks -= 0x1000
#     mid = ticks & 0xff0;   if mid: emit E<mid:03x>
#     low = ticks & 0xf;     if low: emit E<low:x>
#
# so a run is  BIG^n [MID] [LOW]  with n = delta // 0x1000, mid = (delta % 0x1000) & 0xff0,
# low = delta & 0xf. Every delta has exactly ONE run, and every state of the walk has exactly one
# prefix — which is what makes the feasibility question below decidable in closed form rather than
# by search, and exhaustively checkable against brute force.

BIG = 0x1000

# Automaton stages. The stage says which tokens may still follow, and (with the accumulated delta)
# uniquely determines the prefix emitted so far.
STAGE_BIG = 0		# only BIG so far; MID, LOW, BIG or end may follow
STAGE_MID = 1		# MID emitted; only LOW or end may follow
STAGE_LOW = 2		# LOW emitted; the run is closed, only end may follow


def elapse_value (tok):
	'''The tick value of an E-prefixed token, or None if `tok` is not one.

	Mirrors Midiseq2Tokenizer._is_elapse plus the parse, since every caller needs both answers and
	asking twice invites them to disagree.
	'''
	if len(tok) < 2 or tok[0] != 'E':
		return None
	try:
		return int(tok[1:], 16)
	except ValueError:
		return None


def elapse_class (value):
	'''Which automaton slot an elapse VALUE occupies: STAGE_BIG for 0x1000, STAGE_MID for a nonzero
	multiple of 0x10 below 0x1000, STAGE_LOW for 0x1..0xf. None for anything else.

	The three classes partition the 271 E tokens of the shipped vocab exactly (1 + 255 + 15), and
	the test asserts that partition rather than trusting this comment.
	'''
	if value == BIG:
		return STAGE_BIG
	if 0 < value < BIG and (value & 0xf) == 0:
		return STAGE_MID
	if 0 < value <= 0xf:
		return STAGE_LOW
	return None


def stage_admits (stage, cls):
	'''May an elapse token of class `cls` follow a run at `stage`? PURE GRAMMAR -- no target interval.

	The automaton above written as a predicate:

	  STAGE_BIG  BIG, MID or LOW may follow      (BIG^n, then at most one MID and one LOW)
	  STAGE_MID  only LOW                        (one MID per run, and no BIG after it)
	  STAGE_LOW  nothing                         (the run is closed)

	This is the half of `feasible_elapse_tokens` that needs NO alignment evidence, and it is separated
	out because it must hold on every run, including one with no adjudicator: `[E1000]* [Exxx]? [Ex]?`
	is the grammar, so `E160 E010` is malformed whatever the rhythm argues. Keeping it here rather
	than re-deriving it at the two call sites is what stops the decoder mask and the feasibility mask
	from disagreeing about what the automaton says.
	'''
	if cls is None:
		return False
	if stage == STAGE_LOW:
		return False
	if stage == STAGE_MID:
		return cls == STAGE_LOW
	return True


def reachable_range (run_delta, stage):
	'''Totals still reachable from an in-run state, as (lo, hi) with hi None meaning unbounded.

	  STAGE_BIG  every integer >= run_delta   (BIG repeats without limit, then any MID and LOW)
	  STAGE_MID  [run_delta, run_delta + 0xf] (only a LOW may still be added)
	  STAGE_LOW  exactly run_delta            (the run is closed)
	'''
	if stage == STAGE_BIG:
		return run_delta, None
	if stage == STAGE_MID:
		return run_delta, run_delta + 0xf
	return run_delta, run_delta


def feasible_elapse_tokens (run_delta, stage, delta_lo, delta_hi, e_values):
	'''Which elapse tokens may follow this state without leaving [delta_lo, delta_hi] unreachable.

	`e_values` maps token id (or token string — the caller's key type is returned unchanged) to its
	tick value, so the caller keeps its own id space. Returns (allowed, may_end):

	  allowed   the subset of `e_values` keys that keeps a total inside the interval reachable
	  may_end   whether the run may CLOSE here, i.e. whether run_delta is already in the interval.
	            The caller uses this to decide whether non-elapse tokens (a keyword) are admissible;
	            it is half the mask, and forgetting it would let a run end on an infeasible total.

	Closed form, per token class, from `reachable_range` of the state each one leads to:

	  BIG     -> (d + 0x1000, STAGE_BIG): reachable [d+0x1000, inf)  feasible iff hi >= d+0x1000
	  MID m   -> (d + m, STAGE_MID):      reachable [d+m, d+m+0xf]   feasible iff d+m <= hi and
	                                                                  d+m+0xf >= lo
	  LOW l   -> (d + l, STAGE_LOW):      reachable {d+l}            feasible iff lo <= d+l <= hi

	BIG and MID require STAGE_BIG (one MID per run, and no BIG after it); LOW is admissible from
	STAGE_BIG too, since a delta with no mid nibble-byte renders as BIG^n LOW directly (0x1004 ->
	`E1000 E4`). Nothing may follow STAGE_LOW.
	'''
	allowed = set()
	if delta_hi is not None and delta_hi < delta_lo:
		return allowed, False		# empty interval: the caller must fall back, not stall
	may_end = delta_lo <= run_delta and (delta_hi is None or run_delta <= delta_hi)
	if stage == STAGE_LOW:
		return allowed, may_end
	for key, value in e_values.items():
		cls = elapse_class(value)
		if cls is None:
			continue
		if cls == STAGE_BIG:
			if stage == STAGE_BIG and (delta_hi is None or run_delta + BIG <= delta_hi):
				allowed.add(key)
		elif cls == STAGE_MID:
			if stage != STAGE_BIG:
				continue
			total = run_delta + value
			if (delta_hi is None or total <= delta_hi) and total + 0xf >= delta_lo:
				allowed.add(key)
		else:
			total = run_delta + value
			if delta_lo <= total and (delta_hi is None or total <= delta_hi):
				allowed.add(key)
	return allowed, may_end


def canonical_run (delta):
	'''Delta -> its canonical elapse token VALUES. Mirrors Midiseq2Tokenizer._elapse, in value space.

	Kept here so the exhaustive test can brute-force the feasibility predicate without importing the
	tokenizer's string rendering; `tests/midi/align_check.py` also asserts the two agree.
	'''
	out = []
	ticks = delta
	while ticks >= BIG:
		out.append(BIG)
		ticks -= BIG
	mid = ticks & 0xff0
	if mid:
		out.append(mid)
	low = ticks & 0xf
	if low:
		out.append(low)
	return out


# --- grammar walk -------------------------------------------------------------------------

# Token classes for the walk. A midiseq2 line is `[elapse-run] keyword args`, and an event's END is
# NOT determined by counting its arguments -- no arity table exists anywhere in this pipeline, and
# note_on_events() deliberately does without one. It is determined by what comes NEXT: an elapse
# token, another keyword, or a special all close the open event. So there is no decidable boolean
# "this position is an elapse position"; there is only "may an elapse token appear here, and if a
# run is already open, what has it accumulated". Those two are what GrammarState answers.
CLS_NONE = 'none'
CLS_ELAPSE = 'elapse'
CLS_KEYWORD = 'keyword'
CLS_FIELD = 'field'
CLS_SPECIAL = 'special'
# A BARE hex digit (or `_`), which is an argument DIGIT rather than a typed field: `set_tempo 7 a 1 2
# 0`, `ticks_per_beat 1 e 0`. Split out of CLS_FIELD because its legal predecessors are a strictly
# smaller set than a typed field's -- see `bare_allowed`.
CLS_BARE = 'bare'
# The channel token `C1`..`Cf`. Also split out, because it is the one non-keyword that a bare digit
# may follow (`pitchwheel C4 2 0 0 0`), MEASURED at 849 of 8936 bare tokens.
CLS_CHANNEL = 'channel'

_BARE_CHARS = frozenset('0123456789abcdef_')


def token_class (tok, keywords):
	'''Classify one token string. `keywords` is the event-keyword set (keyword_tokens()).'''
	if tok.startswith('<'):
		return CLS_SPECIAL
	if elapse_value(tok) is not None:
		return CLS_ELAPSE
	if tok in keywords:
		return CLS_KEYWORD
	if len(tok) == 1 and tok in _BARE_CHARS:
		return CLS_BARE
	if len(tok) == 2 and tok[0] == 'C' and tok[1] in _BARE_CHARS and tok[1] != '_':
		return CLS_CHANNEL
	return CLS_FIELD


# What may legally FOLLOW an elapse token. A positive allowlist, not a ban list, because the evidence
# is exhaustive in that direction: over 1,316,346 elapse-to-next transitions in the 507-file corpus,
# the successor was a KEYWORD (68.7%) or another ELAPSE (31.3%) and NOTHING else -- no bare digit, no
# #pitch, no $vel, no channel. Those would each be an argument with no event to belong to (`E140 #4c`,
# `E140 C4`). CLS_SPECIAL is admitted on structural grounds rather than corpus evidence: these source
# files carry no @measure, so <eom> never appears in them, but <eom>/<eos> close a measure or the
# piece and time may certainly have passed first (the model's own argmax after an elapse run was
# <eom> at one measured position).
ELAPSE_SUCCESSORS = frozenset({CLS_ELAPSE, CLS_KEYWORD, CLS_SPECIAL})

# What a BARE digit may follow: a keyword whose argument it is (`set_tempo 7`), another digit in the
# same argument (`7 a 1 2 0`), or a channel token (`pitchwheel C4 2`). MEASURED over 8936 bare tokens:
# BARE 6357, KEYWORD 1730, CHANNEL 849, everything else 0.
BARE_PREDECESSORS = frozenset({CLS_BARE, CLS_KEYWORD, CLS_CHANNEL})


class GrammarState:
	'''Where the elapse automaton stands, maintained incrementally over committed tokens only.

	Deliberately NOT derived from logits. The mask exists to correct the model's distribution, so
	deciding whether to apply it from that same distribution is circular — and it fails exactly
	where it matters, since the defect being targeted (a confidently wrong bar length) is a case of
	the model putting its mass on the wrong token. A grammar state is causal, deterministic, and
	free: it depends only on what has already been emitted, so beam_size=1 reproduces greedy
	byte-for-byte.

	`elapse_allowed` is the one structural rule the walk enforces: an event needs at least one
	argument, so an elapse token may not directly follow a keyword. Everything else may be followed
	by one.

	This tracks a WITHIN-RUN accumulator that resets at each event, not a second global tick
	counter — translateMidiseq2's own note_on_events remains the only authority on absolute tick.
	The two are checked against each other in tests/midi/align_check.py (sum of completed run
	deltas == that walk's accumulated tick), because coexisting counters that are never compared
	are how bar lines drift away from the notes they bound.
	'''

	__slots__ = ('prev_class', 'run_delta', 'stage', 'open_keyword', 'total_delta')

	def __init__ (self):
		self.prev_class = CLS_NONE
		self.run_delta = 0
		self.stage = STAGE_BIG
		self.open_keyword = None
		self.total_delta = 0		# sum of every COMPLETED run; compared against note_on_events' tick

	def clone (self):
		'''A detached copy, for beam branching.'''
		out = GrammarState()
		out.prev_class = self.prev_class
		out.run_delta = self.run_delta
		out.stage = self.stage
		out.open_keyword = self.open_keyword
		out.total_delta = self.total_delta
		return out

	@property
	def in_run (self):
		'''True while an elapse run is open (at least one E token emitted, not yet closed).'''
		return self.prev_class == CLS_ELAPSE

	@property
	def elapse_allowed (self):
		'''May ANY elapse token appear at this position?

		Two rules, and it used to enforce only the first:

		  - Not directly after a keyword (`note_on E10` is malformed — the event owes an argument).
		    Yes at a boundary and after a field: `note_on #3b` may legitimately continue with `$22`
		    OR end there and start a delta, and only the model can say which.
		  - Not after a run has closed. A run is `[E1000]* [Exxx]? [Ex]?`, so once a LOW is emitted
		    nothing may follow; at STAGE_MID only a LOW may. Ignoring the stage let the decoder emit
		    `E160 E010` and `E1 E1 E1 E1 E1 E1` -- MEASURED at 10 of 23 runs on a --rank align dump,
		    while the model's own top choice at those positions was a LEGAL token at logprob ~-0.0000.

		`admits` is the per-token form and is what a mask needs; this stays a boolean because callers
		use it to decide whether an elapse BRANCH exists at all.
		'''
		if self.prev_class == CLS_KEYWORD:
			return False
		return self.stage != STAGE_LOW if self.in_run else True

	def admits (self, value):
		'''May the elapse token with tick `value` follow here? The mask's per-token predicate.

		Outside an open run every class is admissible (a fresh run starts at STAGE_BIG); inside one,
		`stage_admits` decides. An off-automaton value is never admissible -- it has no canonical
		prefix, so no legal run contains it.
		'''
		if self.prev_class == CLS_KEYWORD:
			return False
		cls = elapse_class(value)
		if cls is None:
			return False
		return stage_admits(self.stage, cls) if self.in_run else True

	@property
	def bare_allowed (self):
		'''May a BARE hex digit appear here?

		Only as the argument of something that takes one: right after its keyword (`set_tempo 7`),
		continuing an argument already started (`7 a 1 2 0`), or after a channel (`pitchwheel C4 2`).
		Never after an elapse token, which is the case that reached the output as `E1a0 5 5 #53` --
		a digit there belongs to no event at all.
		'''
		return self.prev_class in BARE_PREDECESSORS

	def admits_class (self, cls, value=None):
		'''May a token of class `cls` follow here? The general per-token gate the decoder mask uses.

		Three rules, in the order they bind:

		  - After an ELAPSE token only `ELAPSE_SUCCESSORS` may come. A run is time, and the only
		    things that may follow time are more time, the event that time was leading to, or the end
		    of the measure.
		  - An ELAPSE token additionally answers to the run automaton, via `admits`.
		  - A BARE digit answers to `bare_allowed`.

		Anything not named by a rule is admitted: the mask exists to remove what the grammar forbids,
		not to whitelist a generation policy, and a position this class cannot decide belongs to the
		model.
		'''
		if self.in_run and cls not in ELAPSE_SUCCESSORS:
			return False
		if cls == CLS_ELAPSE:
			return self.admits(value)
		if cls == CLS_BARE:
			return self.bare_allowed
		return True

	def feed (self, tok, keywords):
		'''Commit one token string, advancing the state. Returns its class.'''
		cls = token_class(tok, keywords)
		if cls == CLS_ELAPSE:
			value = elapse_value(tok)
			cls_slot = elapse_class(value)
			if not self.in_run:
				self.run_delta = 0
				self.stage = STAGE_BIG
			self.run_delta += value
			# An off-automaton value (not in the vocab's three classes) would leave `stage` alone;
			# clamp to STAGE_LOW so the run closes rather than accepting further tokens on a state
			# no canonical prefix can produce. A token the stage does not admit gets the same
			# treatment: assigning its class outright let `E140 E1000` walk MID -> BIG and `E5 E140`
			# LOW -> MID, REOPENING a closed run so the next illegal token looked legal. feed accepts
			# whatever it is given (it must be able to walk a malformed stream, e.g. to report on one),
			# but it may only ever move the stage FORWARD.
			if cls_slot is None or not stage_admits(self.stage, cls_slot):
				self.stage = STAGE_LOW
			else:
				self.stage = cls_slot
			self.prev_class = cls
			self.open_keyword = None
			return cls
		# any non-elapse token closes an open run: its accumulated delta is now spent
		if self.in_run:
			self.total_delta += self.run_delta
			self.run_delta = 0
			self.stage = STAGE_BIG
		if cls == CLS_KEYWORD:
			self.open_keyword = tok
		elif cls == CLS_SPECIAL:
			# <eom> carries no time of its own and closes any open event, exactly as the note walk
			# treats it; <bos>/<sep>/<eos> are structural.
			self.open_keyword = None
		self.prev_class = cls
		return cls


def walk_grammar (tokens, keywords, state=None):
	'''Feed a token-string sequence through GrammarState. Returns the state (fresh or advanced).

	Resumable by design, like note_on_events: pass the returned state back to continue, so walking
	in chunks equals walking whole. Beam generation commits one token at a time, so this is the
	only mode that is actually used — the whole-stream form exists for the test.
	'''
	if state is None:
		state = GrammarState()
	for tok in tokens:
		state.feed(tok, keywords)
	return state


# --- soft index ---------------------------------------------------------------------------

def soft_delta (interval_ticks, simultaneous=None):
	'''One softIndex step: tanh(interval / SIMULTANEOUS_TICKS).

	Matcher's normalizeInterval, in ticks instead of milliseconds. Both saturations matter here:
	near zero it collapses a chord onto ONE position (so a 4-note chord in the source is not four
	positions the target has to match one at a time), and at the top it clamps a long rest to 1 (so
	a fermata cannot push the coordinate away without bound).
	'''
	scale = Config['SIMULTANEOUS_TICKS'] if simultaneous is None else simultaneous
	return math.tanh(max(0.0, interval_ticks) / max(scale, 1e-6))


def soft_indices (onsets, simultaneous=None):
	'''Absolute onset ticks -> softIndex positions, starting at 0.'''
	out = []
	position = 0.0
	prev = None
	for tick in onsets:
		if prev is not None:
			position += soft_delta(tick - prev, simultaneous)
		out.append(position)
		prev = tick
	return out


# --- anchor vote --------------------------------------------------------------------------

def gaussian_density (x, sigma):
	'''Unnormalised-by-sigma gaussian, matching mathex.gaussianDensity's use in gaussianSmooth.'''
	return math.exp(-0.5 * (x / sigma) ** 2)


def gaussian_smooth (votes, sigma):
	'''Smooth a sparse {value: weight} histogram over its OWN keys (no grid), as mathex does.

	On its own keys rather than a grid because the offsets are continuous and arbitrary; inventing a
	grid would impose a resolution the data does not have.
	'''
	keys = list(votes.keys())
	return {x: sum(gaussian_density(k - x, sigma) * v for k, v in votes.items()) for x in keys}


def anchor_from_votes (votes, sigma=None, support_frac=None):
	'''Smoothed histogram -> (mode, confidence, lo, hi).

	This is HeuristicNavigator.zeroPosition's scheme, and the reason it is a vote rather than "the
	offset of the best recent match" is robustness: a couple of outlier pairings cannot pull the
	mode, whereas they can trivially be the single best match.

	`confidence` is the mode's share of the smoothed mass, and (lo, hi) spans every offset whose
	density clears `support_frac` of the peak. That span — not a constant tolerance — is what the
	mask width is derived from, so a flat (uncertain) histogram yields a wide, barely-constraining
	interval and a sharp one yields a tight interval. A fixed tolerance is exactly what would make
	the mask rigid.
	'''
	if not votes:
		return None, 0.0, None, None
	sigma = Config['AnchorSigma'] if sigma is None else sigma
	frac = Config['AnchorSupportFrac'] if support_frac is None else support_frac
	smoothed = gaussian_smooth(votes, sigma)
	total = sum(smoothed.values())
	mode = max(smoothed, key=lambda k: smoothed[k])
	peak = smoothed[mode]
	support = [k for k, v in smoothed.items() if v >= peak * frac]
	confidence = (peak / total) if total > 0 else 0.0
	return mode, confidence, min(support), max(support)


# --- alignment state ----------------------------------------------------------------------

class AlignState:
	'''Per-beam alignment between the source window and the generated target stream.

	Carries Matcher's two SEPARATE accumulators, and the separation is the substance rather than
	bookkeeping:

	  cost   decayed (x0.6 per step), bounded by tanh, so it measures how INCONSISTENT the recent
	         alignment is. Local by construction — old errors fade out, which is what makes this
	         usable for ranking a beam mid-piece instead of relitigating the whole file.
	  value  undecayed, +1 per well-matched note, so it measures how much EVIDENCE has accumulated.

	`prior` combines them the way Matcher's priorByOffset does. Both terms saturate, so no beam can
	win on length alone and none can be killed by one bad note.

	Cheap to clone, because beam search clones it per branch and the model forward pass must stay
	the dominant cost.
	'''

	__slots__ = ('src_events', 'src_by_pitch', 'pairs', 'cost', 'value', 'misses', 'matched',
		'last_offset', 'tgt_count', 'ratio', 'ratio_pairs', 'residual', 'residual_n',
		'seed_offset', 'used')

	def __init__ (self, src_events=None, seed_offset=None):
		# src_events: list of dicts with at least onset/pitch/softIndex, in stream order
		self.src_events = []
		self.src_by_pitch = {}
		self.pairs = []			# (src_index, tgt_softIndex, src_softIndex, weight)
		# src_index -> how many times THIS lineage has paired it. Derivable from `pairs`, kept
		# separately because it is read once per candidate per beam per position and a scan of pairs
		# there would make the per-note work grow with the length of the piece.
		self.used = {}
		self.cost = 0.0
		self.value = 0.0
		self.misses = 0
		self.matched = 0
		self.last_offset = None
		self.tgt_count = 0
		self.ratio = None		# tgt ticks per src tick, EMA
		self.ratio_pairs = []		# (src_softIndex, src_tick, tgt_tick) for the baseline slope
		self.residual = None		# EMA of |tick prediction residual|
		self.residual_n = 0
		# Cold-start prior for the candidate window, used ONLY until the vote has pairs to draw on.
		# Sliding inference does not need it: the source WINDOW is itself the bound, so a generated
		# note can only match the few hundred tokens on screen. Whole-file alignment has no such
		# bound — every pitch recurs dozens of times across a piece — so the first note would be free
		# to match anywhere. None keeps the old behaviour (any same-pitch note is a candidate).
		self.seed_offset = seed_offset
		if src_events:
			self.set_source(src_events)

	def set_source (self, events):
		'''Install the source window's note_on events (already carrying softIndex).

		`src_by_pitch` is Matcher's pitchMap: pitch -> the indices carrying it. Candidate generation
		is a lookup rather than a scan, which is what keeps the per-note work independent of the
		window length.
		'''
		self.src_events = list(events)
		self.src_by_pitch = {}
		for i, e in enumerate(self.src_events):
			self.src_by_pitch.setdefault(e['pitch'], []).append(i)

	def clone (self):
		out = AlignState()
		# source is shared: it is read-only for the life of a step, and copying it per branch would
		# dominate the clone cost for no benefit
		out.src_events = self.src_events
		out.src_by_pitch = self.src_by_pitch
		out.pairs = list(self.pairs)
		out.used = dict(self.used)
		out.cost = self.cost
		out.value = self.value
		out.misses = self.misses
		out.matched = self.matched
		out.last_offset = self.last_offset
		out.tgt_count = self.tgt_count
		out.ratio = self.ratio
		out.ratio_pairs = list(self.ratio_pairs)
		out.residual = self.residual
		out.residual_n = self.residual_n
		out.seed_offset = self.seed_offset
		return out

	# --- anchor ------------------------------------------------------------------------

	def votes (self, tgt_softindex):
		'''Ballots from pairs within AnchorSoftSpan of `tgt_softindex`, keyed by offset.'''
		span = Config['AnchorSoftSpan']
		out = {}
		for _, tsi, ssi, weight in reversed(self.pairs):
			if tgt_softindex - tsi > span:
				break
			offset = ssi - tsi
			out[offset] = out.get(offset, 0.0) + weight
		return out

	def anchor (self, tgt_softindex):
		'''(mode, confidence, lo, hi) of the offset histogram at this target position.'''
		return anchor_from_votes(self.votes(tgt_softindex))

	# --- observing a generated note ----------------------------------------------------

	def candidates (self, pitch, tgt_softindex):
		'''Source indices that could be this generated note, as (src_index, offset).

		Two filters, both from Matcher: same pitch (pitchMap), and an offset within
		AnchorSoftSpan of the anchor. The second is the bounded search window — Matcher's SkipDeep
		in offset space rather than index space — and it is what stops candidate generation from
		growing with the source window.

		With no anchor yet every same-pitch note is a candidate unless `seed_offset` supplies a prior;
		there is otherwise nothing to narrow it with, and pretending otherwise would just apply the
		defaults as if measured.
		'''
		hits = self.src_by_pitch.get(pitch)
		if not hits:
			return []
		mode, _confidence, _lo, _hi = self.anchor(tgt_softindex)
		if mode is None:
			mode = self.seed_offset
		span = Config['AnchorSoftSpan']
		out = []
		for i in hits:
			offset = self.src_events[i]['softIndex'] - tgt_softindex
			if mode is None or abs(offset - mode) <= span:
				out.append((i, offset))
		return out

	def observe (self, pitch, tgt_tick, tgt_softindex):
		'''Fold one generated note_on into the alignment. Returns a per-note detail dict.

		The DP relaxation, restricted to the candidate window: take the lowest-cost candidate under
		Matcher's asymmetric offset penalty, then run its cost/value recursion. An unmatched note is
		charged MissCost and earns no value — it is not fatal, because the score arm may legitimately
		carry a note the irregular arm does not have near this position.

		Returns dict(src, self_cost, offset, skip, cost, prior), with `src` None for a miss. The
		per-note `self_cost` is returned rather than only folded into the running cost because it is
		the only quantity attributable to THIS note: `cost` is a decayed sum over the recent past, so
		colouring a note by it would paint the neighbourhood's history onto one note.
		'''
		self.tgt_count += 1
		cands = self.candidates(pitch, tgt_softindex)
		prev_offset = self.last_offset
		best = None
		for index, offset in cands:
			if prev_offset is None:
				self_cost = 0.0
			else:
				bias = offset - prev_offset
				coeff = Config['UnderCost'] if bias > 0 else Config['OverCost']
				self_cost = (bias * coeff) ** 2
			skip = 0
			if self.pairs:
				skip = max(0, index - self.pairs[-1][0] - 1)
			# How many times this lineage has ALREADY paired this source note. 0 for a fresh note --
			# including one arriving out of index order, which is a chord and is free.
			reuse = self.used.get(index, 0)
			total = (self.cost * Config['CostStepAttenuation']
				+ math.tanh(skip * Config['SkipCost'])
				+ math.tanh(reuse * Config['ReuseCost'])
				+ math.tanh(self_cost * Config['SelfCostScale']))
			if best is None or total < best[0]:
				best = (total, index, offset, self_cost, skip, reuse)
		if best is None:
			self.misses += 1
			self.cost = self.cost * Config['CostStepAttenuation'] + Config['MissCost']
			return dict(src=None, self_cost=None, offset=None, skip=0, reuse=0, cost=self.cost,
				prior=self.prior)
		total, index, offset, self_cost, skip, reuse = best
		self.cost = total
		self.value += 1.0 - math.tanh(self_cost * Config['SelfCostScale'])
		self.last_offset = offset
		self.matched += 1
		# weight the ballot by how consistent this pairing was, so a strained match votes weakly
		weight = 1.0 - math.tanh(self_cost * Config['SelfCostScale'])
		self.pairs.append((index, tgt_softindex, self.src_events[index]['softIndex'], weight))
		self.used[index] = self.used.get(index, 0) + 1
		self._update_ratio(self.src_events[index]['softIndex'],
			self.src_events[index]['onset'], tgt_tick)
		return dict(src=index, self_cost=self_cost, offset=offset, skip=skip, reuse=reuse,
			cost=self.cost, prior=self.prior)

	# --- tempo ratio and the tick interval ---------------------------------------------

	def _update_ratio (self, src_softindex, src_tick, tgt_tick):
		'''EMA the tgt/src tick ratio over a baseline at least RatioBaselineSI apart.

		Follower1.updateSpeed's scheme, and the baseline requirement is the whole point: a slope
		taken between two adjacent onsets is dominated by quantisation and rubato, so it walks back
		until the pair is far enough apart in softIndex to carry signal. The residual EMA is
		maintained here too, because the mask's width comes from it.
		'''
		# residual of the CURRENT prediction, before this pair updates the ratio -- otherwise the
		# ratio would be fitted to the point it is then scored on
		predicted = self.predict_tick(src_tick)
		if predicted is not None:
			error = abs(tgt_tick - predicted)
			self.residual = (error if self.residual is None
				else error * Config['RatioEmaAlpha'] + self.residual * (1 - Config['RatioEmaAlpha']))
			self.residual_n += 1
		self.ratio_pairs.append((src_softindex, src_tick, tgt_tick))
		last = self.ratio_pairs[-1]
		for base in reversed(self.ratio_pairs[:-1]):
			if last[0] - base[0] > Config['RatioBaselineSI']:
				d_src = last[1] - base[1]
				d_tgt = last[2] - base[2]
				if d_src > 0:
					slope = d_tgt / d_src
					if slope > 0:
						self.ratio = (slope if self.ratio is None
							else slope * Config['RatioEmaAlpha']
								+ self.ratio * (1 - Config['RatioEmaAlpha']))
				break

	def predict_tick (self, src_tick):
		'''Where the target should be when the source is at `src_tick`, or None with no ratio yet.

		Anchored on the most recent pair rather than on the stream origin, so an inherited offset is
		not re-charged at every prediction — the same reason the accuracy harness reports a rebased
		span_f1 alongside the raw absolute-tick onset_f1.
		'''
		if self.ratio is None or not self.ratio_pairs:
			return None
		_si, base_src, base_tgt = self.ratio_pairs[-1]
		return base_tgt + self.ratio * (src_tick - base_src)

	def tick_interval (self, src_tick):
		'''Feasible absolute target tick interval for a note aligned to `src_tick`.

		(None, None) when there is not enough evidence — too few pairs, no ratio, or no residual
		history — and that is the honest answer rather than a default-width guess: a mask derived
		from unmeasured constants is worse than no mask, because its mis-kills are invisible.

		The half-width is RatioResidualK times the EMA of the model's OWN recent prediction error,
		floored. So the interval is exactly as wide as the model has recently been wrong, which is
		what keeps this from being rigid: while the model tracks the source closely the mask tightens
		around it, and as soon as it starts drifting the mask opens up instead of fighting it.
		'''
		if len(self.pairs) < Config['MinPairsForMask']:
			return None, None
		predicted = self.predict_tick(src_tick)
		if predicted is None or self.residual is None:
			return None, None
		half = max(Config['RatioResidualK'] * self.residual, Config['RatioResidualFloor'])
		return predicted - half, predicted + half

	# --- forecasting a tick, before its pitch exists -----------------------------------

	def unmatched_ahead (self, limit):
		"""Source indices not yet paired, from the last pairing forward, at most `limit` of them.

		"Ahead" is in SOURCE STREAM ORDER, not in tick order around a predicted position: the
		alignment is monotone by construction (`skip` charges for jumping over source notes), so the
		notes this target can still legitimately claim are the ones after the last one it claimed.
		Taking them in tick order instead would offer the forecast notes it has already passed.
		"""
		used = {index for index, _tsi, _ssi, _w in self.pairs}
		start = (self.pairs[-1][0] + 1) if self.pairs else 0
		out = []
		for i in range(start, len(self.src_events)):
			if i in used:
				continue
			out.append(i)
			if len(out) >= limit:
				break
		return out

	def forecast (self, tgt_tick, lookahead=None):
		"""What the alignment would cost if the next onset landed at `tgt_tick`, pitch unknown.

		The problem this solves: an elapse token decides a tick, but the pitch that would let
		`observe` score it is one or two tokens away, so at the position where the rhythm is actually
		chosen the aligner has nothing to say. MEASURED on a width-4 dump: of 554 elapse candidates,
		0 carried an alignment verdict -- the decision that fixes the rhythm was taken on the language
		model alone, and the model's elapse distribution is high-entropy (top-4 spanning 1.30 to 2.50
		nats on the case that motivated this).

		This is scored in TICK space, on `predict_tick`, and NOT through `observe`. That is not an
		implementation preference, it is forced: `observe`'s self_cost is a function of the softIndex
		offset, and soft_delta saturates (tanh(320/115) = 0.992, tanh(960/115) = 1.000), so every
		elapse candidate past ~350 ticks lands on the same softIndex and an observe-based forecast is
		FLAT across exactly the choices that need separating. Measured on the motivating case: 320,
		480, 720 and 960 ticks all forecast 0.3963 through observe, while in tick space they are 226,
		66, 174 and 414 ticks from the prediction. softIndex is the right instrument for a pitch
		verdict (which note was matched) and the wrong one for a rhythm verdict (which tick).

		The error is normalised by the mask's own half-width -- RatioResidualK times the EMA of the
		model's recent prediction error -- so it is dimensionless, 1.0 at the band edge, and as
		forgiving as the model has recently been inaccurate. Then bounded (by `x/(1+x)`, not tanh --
		see below) and folded into the attenuated running cost, the same recursion `observe` uses, so
		a forecast and a verdict live on one scale and the pool can be sorted on either.

		Returns (cost, src_index, predicted_tick), or (None, None, None) when the aligner has NO
		OPINION -- no ratio yet, no residual history, or nothing left unmatched ahead. That is the
		honest answer and the caller must fall back to the language model rather than read the
		absence as a judgement; a forecast derived from unmeasured defaults would be worse than none,
		because its mis-rankings would be invisible.
		"""
		limit = Config['ForecastLookahead'] if lookahead is None else lookahead
		ahead = self.unmatched_ahead(limit)
		if not ahead or self.ratio is None or self.residual is None:
			return None, None, None
		half = max(Config['RatioResidualK'] * self.residual, Config['RatioResidualFloor'])
		first = ahead[0]
		best = None
		for i in ahead:
			predicted = self.predict_tick(self.src_events[i]['onset'])
			if predicted is None:
				continue
			# Optimistic across the lookahead: this tick may legitimately be aiming at any upcoming
			# onset, and it must not be charged for choosing the wrong one before a pitch exists. But
			# reaching a LATER one means jumping the ones between, so charge for that exactly as
			# `observe` does -- without this term a far tick can win by aiming past the notes it
			# skipped (measured: E3c0 at 3200 ranked 2nd by aiming at source #14 and ignoring #10-13).
			err = abs(tgt_tick - predicted) / half
			# `x / (1 + x)` rather than tanh, and the reason is numerical rather than aesthetic: both
			# are bounded by 1, but tanh is FLAT IN FLOAT past about 3 -- tanh(50) and tanh(100) are
			# both exactly 1.0 -- so with a narrow band (a model tracking the source closely floors
			# the half-width at RatioResidualFloor = 24 ticks) every candidate more than ~2 bands out
			# ranks equal and the forecast silently stops discriminating. This form is strictly
			# increasing everywhere and still distinguishes 50 from 100, which is exactly the range an
			# elapse candidate can span. Same shape for the skip term, for the same reason.
			skip = max(0, i - first) * Config['SkipCost']
			term = err / (1.0 + err) + skip / (1.0 + skip)
			if best is None or term < best[0]:
				best = (term, i, predicted)
		if best is None:
			return None, None, None
		term, index, predicted = best
		cost = self.cost * Config['CostStepAttenuation'] + term
		return cost, index, predicted

	# --- ranking -----------------------------------------------------------------------

	@property
	def prior (self):
		'''Matcher's bounded evidence-minus-inconsistency score, for ordering beams.

		Bounded on both sides (each term is a tanh), which is what makes a single lambda weight it
		against a length-normalised log-probability without retuning per piece. An unbounded
		alignment term would need lambda re-fitted for every file length.
		'''
		return (math.tanh(self.value * Config['PriorValue'])
			- math.tanh(self.cost * Config['PriorCost']))

	def report (self):
		'''Instrumentation snapshot. The mask can only ever delete the right answer, so every field
		a caller needs to detect that is exposed here rather than inferred from the output.'''
		return dict(matched=self.matched, misses=self.misses, tgt_count=self.tgt_count,
			cost=self.cost, value=self.value, prior=self.prior,
			ratio=self.ratio, residual=self.residual, pairs=len(self.pairs))
