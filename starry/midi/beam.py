'''Beam search over a midiseq2 token stream, branching only where the choice can change the music.

Separated from tools/midi/translateMidiseq2.py on purpose. That script is the measured, working
greedy translator; this is a search layered on top of it, and the two have different reasons to
change. Nothing here imports it, so a beam bug cannot reach a greedy run.

The search is generic over a step function, so one copy of the bookkeeping serves both the
decoder-only and the encoder-decoder translator. The bookkeeping is where the interesting bugs live
(off-by-one positions, a beam that finishes and takes its parent's state with it, a pool that ranks
hypotheses of unequal length against each other), so there is exactly one of it.

  BRANCH POLICY   only elapse and pitch positions fan out; everywhere else the beam takes its own
                  argmax. A branch has to pay for itself: the batch is B wide for the whole step, so
                  widening at a position whose choice cannot change the alignment spends B forward
                  passes to re-derive one answer.

                  This is NOT very selective, and the number is worth stating rather than implying:
                  MEASURED 86.2% of positions branch (114349 positions over 12 test files; elapse
                  69815, pitch 28784, none 15750). An elapse token is legal almost everywhere -- the
                  grammar forbids it only straight after a keyword -- so "elapse is possible here" is
                  a weak filter. The velocity rule below is what took it down from 99.7%.

                  What actually bounds the cost is therefore beam_size, not the policy: the pool is
                  capped at beam_size survivors per position, so width B costs ~B forward-batched
                  rows regardless of how often it branches. Measured 3.7x greedy wall-clock at width
                  4 on CPU. The policy's remaining value is that it keeps the batch at B rather than
                  B*branch_k, and that pitch/elapse are where the alignment can eventually adjudicate
                  -- not that it makes the search cheap.
  RANKING         raw logprob within a step (every live beam is the same length there, so
                  normalising cannot reorder them), length-normalised only at the end between
                  FINISHED hypotheses, which do differ in length.
  DETERMINISM     no temperature, no sampling. beam_size=1 reproduces greedy argmax exactly, which
                  tests/midi/beam_parity_check.py asserts byte-for-byte against the real thing
                  rather than reasoning about it.

Where a branch point IS is decided by the grammar walk in starry/midi/align.py, never by the
logits. The failure being targeted is the model putting its mass confidently on a wrong delta, and a
confidence test does not flag exactly that case -- so consulting the distribution to decide where to
correct the distribution would be circular AND would fail where it matters.
'''

import itertools
import math

from .align import (GrammarState, CLS_ELAPSE, CLS_KEYWORD, CLS_SPECIAL, CLS_BARE, STAGE_MID,
	elapse_value, token_class)


BRANCH_NONE = 0
BRANCH_ELAPSE = 1
BRANCH_PITCH = 2

BRANCH_NAMES = {BRANCH_NONE: 'none', BRANCH_ELAPSE: 'elapse', BRANCH_PITCH: 'pitch'}

# Events that take a pitch argument. This is a GRAMMAR fact (it decides where `pitch_pending`
# holds), not a branch policy -- see BRANCH_PITCH_EVENTS for the policy.
PITCH_EVENTS = ('note_on', 'note_off')

# Events whose pitch argument is a BRANCH POINT. note_off is excluded: its pitch is determined by
# which notes are open, and the alignment never observes a note_off, so neither the model nor the
# aligner has an opinion to search over. MEASURED on a width-4 dump (window 0 of I-YIgmEZ0ss, 200
# positions): 116 note_off-pitch branch points spent 464 of 2741 pool slots (16.9%) and a non-argmax
# note_off pitch survived the cut 0 times, against 10 survivals for note_on pitch. The model is also
# near-certain there (mean top-1 logprob -0.0711, >-0.05 at 75% of them) with the runners-up 8.8 nats
# behind, so those slots were spent re-deriving one answer and displacing real competitors.
#
# Like VELOCITY_AFTER_PITCH this only ever REMOVES a branch point; it bans no token and cannot make
# an output illegal. A wrong note_off pitch leaving a note hanging is a MASK concern (the argument
# the old comment here made), and a mask is where it belongs -- widening cannot fix it, since the
# beam only ever picked the argmax at those positions anyway.
BRANCH_PITCH_EVENTS = ('note_on',)

# Events whose pitch is ALWAYS followed by a `$` velocity, so the position after that pitch is not a
# decision at all. Measured over the test corpus (20 files): note_on is `#$` in 22199/22199 lines and
# note_off is `#` in 22199/22199, while control_change genuinely splits 1597 `#` / 1590 `#$` and so
# stays a branch point. This is the one arity fact strong enough to act on; a fuller arity table is
# derivable but this is the position it would change.
#
# It only ever REMOVES a branch point, never bans a token. If the measurement were wrong the search
# would explore less than it could -- it could not make an output illegal -- which is why a corpus
# statistic is admissible here and would not be in the mask.
VELOCITY_AFTER_PITCH = ('note_on',)


class BranchState:
	'''Grammar position plus the one extra bit the branch policy needs.

	Wraps GrammarState rather than extending it. The elapse automaton is a property of the LANGUAGE
	and is checked exhaustively in tests/midi/align_check.py; "has this event stated its pitch yet"
	is a property of THIS decoding policy. Keeping the second out of the first means the verified
	thing stays verified, and a change of branch policy cannot quietly alter what the mask considers
	reachable.
	'''

	__slots__ = ('grammar', 'pitch_pending', 'velocity_pending', 'saw_note_on',
		'saw_elapse_after_note_on')

	def __init__ (self, grammar=None, pitch_pending=False, velocity_pending=False,
		saw_note_on=False, saw_elapse_after_note_on=False):
		self.grammar = grammar if grammar is not None else GrammarState()
		# True between a note_on/note_off keyword and its `#XX`: exactly the positions where a pitch
		# token is the expected next thing.
		self.pitch_pending = pitch_pending
		# True between a note_on's pitch and its velocity, where nothing is being decided.
		self.velocity_pending = velocity_pending
		# Has this lineage emitted a note_on yet? A LATCH -- once set it never clears, because it marks
		# a phase of the file and not a property of the current position.
		#
		# It is the FIRST half of the align-ranking gate: before the first note_on the file is still in
		# its header and the aligner has nothing to say, so the language model decides alone. MEASURED
		# on the run that motivated this: with the aligner ranking from position 0, `Eb40` (logprob
		# -19.11, the model hates it because it declines to write a header at all) beat `ticks_per_beat`
		# (-0.00000), and the search went straight to the notes -- which scored WELL, since
		# compose_output supplies the source's header as a fallback. Good numbers by a route that is not
		# a translation decision: whether to emit a header is not something the alignment is entitled to
		# an opinion about.
		#
		# Scopes itself to the piece's opening with no cross-window state, because seed_state walks the
		# prefix's target half through `feed`: any window primed mid-piece starts with the latch set.
		self.saw_note_on = saw_note_on
		# Has an elapse token been committed AFTER that first note_on? The SECOND half of the gate, and
		# a latch for the same reason. Ordered strictly after `saw_note_on`, hence the name: an elapse
		# in the header does not count, because the header's elapse decisions are the ones the first
		# half is there to keep the aligner out of.
		#
		# It exists because the aligner's elapse opinion is NOT MERELY unhelpful before the first real
		# tick advance -- it is undefined. `AlignState.forecast` needs a tick ratio, `_update_ratio`
		# fits `slope = d_tgt / d_src` behind an `if slope > 0` guard, and with every target tick still
		# at 0 that guard never passes, so `ratio` stays None and `forecast` returns (None, None, None)
		# forever. MEASURED on the committed dump: the aligner abstained at all 47 elapse branches on
		# the winning chain, and all 37 candidates at the first post-note_on elapse branch carried
		# `loss None`. Asking it there produced nothing but the appearance of having been consulted.
		#
		# So the gate now opens only once the tick has actually moved, which is exactly the condition
		# under which the ratio can exist. Note this makes the FIRST elapse after the header the model's
		# alone by construction, which is intended: it is the observation that bootstraps the ratio, and
		# there is nothing yet for the aligner to have an opinion with.
		self.saw_elapse_after_note_on = saw_elapse_after_note_on

	def clone (self):
		return BranchState(self.grammar.clone(), self.pitch_pending, self.velocity_pending,
			self.saw_note_on, self.saw_elapse_after_note_on)

	def branch_kind (self):
		'''What kind of branch point the NEXT position is, from committed tokens only.

		A pitch is pending only immediately after note_on/note_off, and an elapse token is legal
		anywhere except directly after a keyword (an event owes at least one argument), so the two
		cases cannot both hold and the order of these tests is not a tie-break.
		'''
		if self.pitch_pending:
			# Policy, not grammar: a pitch is still PENDING after note_off (the grammar owes one), but
			# only note_on's pitch is worth fanning out on. `open_keyword` is the keyword that opened
			# this event and is still set while its pitch is pending.
			return (BRANCH_PITCH if self.grammar.open_keyword in BRANCH_PITCH_EVENTS
				else BRANCH_NONE)
		if self.velocity_pending:
			# A note_on owes a velocity here and velocity does not enter the alignment at any point,
			# so there is nothing to search: measured 0 of 14392 note_on pitches were followed by an
			# elapse token. Branching here would spend the whole batch re-deriving `$`.
			return BRANCH_NONE
		if not self.grammar.elapse_allowed:
			return BRANCH_NONE
		if self.grammar.in_run and self.grammar.stage == STAGE_MID:
			# STAGE_MID is NOT an elapse branch point. The only elapse token the automaton still admits
			# here is a LOW (0x1..0xf), and a LOW is a 1..15 tick rounding correction against a 480-tick
			# beat -- there is no rhythmic choice left to search, the coarse one was made by the MID
			# behind this position. So this is an ordinary position: no fan-out, no forced proposals, no
			# align ranking. A LOW still reaches the output whenever the model itself puts one at the
			# argmax; it is simply never reinforced.
			#
			# MEASURED before this, on the committed 200-token dump: 109 of 248 proposal-drawing rows
			# were at STAGE_MID, and every one of them spent all 8 slots on lows the model rated 2 to 22
			# nats below its own best. 82 of the 84 lows that reached the winning path could ONLY have
			# arrived as forced proposals -- the logprob margin would have cut every one.
			return BRANCH_NONE
		return BRANCH_ELAPSE

	def feed (self, tok, keywords):
		'''Commit one token string. Returns its grammar class.'''
		open_kw = self.grammar.open_keyword		# read BEFORE feed clears it
		cls = self.grammar.feed(tok, keywords)
		if tok == 'note_on':
			self.saw_note_on = True
		elif cls == CLS_ELAPSE and self.saw_note_on:
			# `elif` costs nothing (no token is both) but says the thing: these two latches are ordered,
			# and an elapse reaching here with `saw_note_on` still clear is a header elapse that must NOT
			# open the gate.
			self.saw_elapse_after_note_on = True
		if cls == CLS_KEYWORD:
			self.pitch_pending = tok in PITCH_EVENTS
			self.velocity_pending = False
		elif self.pitch_pending and tok.startswith('#'):
			self.pitch_pending = False		# this event has stated its pitch
			self.velocity_pending = open_kw in VELOCITY_AFTER_PITCH
		elif self.velocity_pending:
			# Satisfied by the `$` it was waiting for, or abandoned by anything else -- either way the
			# obligation is discharged and the next position is a real decision again.
			self.velocity_pending = False
		elif cls in (CLS_ELAPSE, CLS_SPECIAL):
			self.pitch_pending = False		# the event closed without one
		return cls


# Lineage identity for an observer. The search itself never reads it -- a beam is defined by its
# ids and its score, not by a name -- but an inspector needs a stable handle to hang per-beam state
# (an alignment, a tick cursor) off, and `id()` is not one: a pruned beam is collected and the next
# allocation can reuse its address, silently grafting one hypothesis's history onto another.
_UID = itertools.count(1)


class Beam:
	'''One hypothesis: its generated tokens, its score, and the state that goes with them.'''

	__slots__ = ('ids', 'logprob', 'state', 'finished', 'forced', 'uid')

	def __init__ (self, ids=None, logprob=0.0, state=None, finished=False, forced=False, uid=None):
		self.uid = next(_UID) if uid is None else uid
		self.ids = ids if ids is not None else []
		self.logprob = logprob			# sum of log p over generated tokens, NOT length-normalised
		self.state = state if state is not None else BranchState()
		self.finished = finished		# saw the terminator; the terminator itself is not kept in ids
		# Kept for a caller that wants to tag a hypothesis, but the SEARCH never sets it: the
		# first-position ban applies to the position, not to any one beam, so it is reported once as
		# beam_search's second return value rather than copied onto every hypothesis.
		self.forced = forced

	def clone (self, uid=None):
		'''A fresh hypothesis with the same tokens. A NEW uid by default, because the caller is about
		to append a token and the result is a new node.

		Pass `uid` to keep an identity: a finished hypothesis appends nothing (the terminator is not
		kept), so it IS its parent's token sequence and must carry the parent's uid, or `best_uid`
		names a node no observer ever saw and every consumer that walks the tree from it -- the
		lineage carry, the viewer's winning path -- comes up empty.
		'''
		return Beam(list(self.ids), self.logprob, self.state.clone(), self.finished, self.forced, uid)

	def score (self, alpha):
		'''Length-normalised score, for comparing hypotheses of DIFFERENT length.

		Only ever used at the end, between finished beams. Within a step every live beam has emitted
		the same number of tokens, so normalising there would divide the whole pool by one constant
		and could not reorder it -- the ranking would be identical and the numbers only harder to
		read against a logprob.
		'''
		if not self.ids:
			return self.logprob
		return self.logprob / (len(self.ids) ** alpha if alpha else 1.0)


def beam_search (step, tokens, keywords, eos_id, max_new, beam_size=4, branch_k=4,
	length_alpha=0.7, seed_state=None, ban_first=(), rank=None, report=None, observer=None,
	greedy_first=True, align_from_first_elapse=True, logprob_margin=2.0,
	adjudicator=None, elapse_k=0, adjudicate=True):
	'''Run the search. Returns (best ids, forced, report).

	step(rows) -> log-probability rows. `rows` is a list of generated-id lists, one per live beam,
	ALL THE SAME LENGTH (a finished beam leaves the pool, so the batch stays rectangular and needs no
	padding -- and therefore no attention mask, which is one fewer thing that can be silently
	wrong). It must return a [len(rows), vocab] float tensor of log-probabilities.

	ban_first: token ids removed from the FIRST generated position. The caller's terminator ban lives
	here. `forced` reports whether the argmax at that position was banned, because the caller keys
	its end-of-piece detection on that having happened -- a ban that hid itself would leave the
	caller unable to tell a finished piece from a stalled one.

	greedy_first: take the model's argmax at the first generated position and do not branch there,
	default True. SAME SCOPE as ban_first -- the first position of each beam_search call, i.e. of each
	window, since `live` is rebuilt from seed_state per call.

	The reason is that the aligner has nothing to say at that position and the model has everything to
	say. MEASURED on the committed align dump's position 0: the argmax was `ticks_per_beat` at
	logprob -0.00000 and the other three beam slots went to `format_type` (-16.77), `#52` (-18.05) and
	`Eb40` (-19.11), while EVERY candidate's loss was None -- AlignState has no pairs yet, so it
	abstains outright. Three of four slots spent on continuations the model rates 17 to 19 nats worse,
	with no alignment evidence to justify the spend, and those lineages then persist and compete for
	slots at later positions where the evidence does exist.

	align_from_first_elapse: hold the adjudicator OFF until a lineage has emitted its first note_on
	AND THEN an elapse token, default True. TWO latches on BranchState, not position tests, so they
	survive the grammar wandering back through header-shaped tokens, and because seed_state replays
	the primer through `feed` they scope to the PIECE's opening rather than to each window's.

	Same reasoning as greedy_first, two steps further, and the two steps are different reasons.

	The note_on half: before the first note_on the file is still its header, and the alignment's only
	opinion there is an accident. With ranking from position 0 the winning lineage descended from that
	`Eb40` at -19.11 -- a token that declines to write a header at all and goes straight to the music,
	reaching scoreable pitches by position 2 and scoring well from there. Well, but by a route the
	aligner is not entitled to choose: `compose_output` supplies the SOURCE's header whenever the body
	carries none, so the good numbers were bought with a decision about the header that no alignment
	evidence bears on.

	The elapse half: after the header but before any tick advance, the aligner's elapse opinion is not
	merely unhelpful, it is UNDEFINED. `forecast` needs a tick ratio; `_update_ratio` fits
	`slope = d_tgt / d_src` behind an `if slope > 0` guard; with every target tick still 0 that guard
	never passes, so `ratio` stays None and `forecast` returns (None, None, None) for the rest of the
	run. MEASURED on the committed dump: the aligner abstained at ALL 47 elapse branches on the winning
	chain, and all 37 candidates at the first post-note_on elapse branch carried `loss None` while the
	model put -0.0000 on elapse-zero against -8.9749 for the cheapest real elapse. Consulting it there
	bought nothing but the appearance of having consulted it -- and, before the enumeration was gated
	too, a pool full of forced candidates the LM then ordered.

	So the gate opens exactly when the ratio can exist. This makes the first elapse after the header
	the model's alone BY CONSTRUCTION, which is the intent: that token is the observation that
	bootstraps the ratio, and until it lands there is nothing for the aligner to have an opinion with.

	Set False to measure all of this (the ablation).

	logprob_margin: drop any candidate rated more than this many nats below its OWN ROW's best,
	default 2.0. None disables it.

	`branch_k` is a fixed width, so without this the search spends the full width at every branch point
	however certain the model is there. MEASURED on the committed align dump's first elapse branch after
	the header: the four rows put -0.0000/-0.0000/-0.0002/-0.0003 on elapse-zero and the next candidate
	was 8.97 nats or worse, so three of four slots per row went to continuations already ruled out.

	Per ROW, not per pool: each live beam keeps its own plausible continuations, so a beam the model
	likes less overall is not silently pruned to a narrower width than its neighbours. A row can never
	be emptied, since the argmax's own margin is 0. And it cannot fire at width 1 (`len(cands) > 1` is
	false), which is what makes greedy/beam-1 byte-identity safe by construction rather than by luck.

	The adjudicator's forced elapse proposals are EXEMPT, because they are appended after this and by
	design sit far outside any useful margin -- the case that motivated the whole adjudicator had the
	correct elapse at -5.36 against the argmax's -0.08, and the dump's cheapest real elapse was 8.97
	nats back. A margin over those would delete the aligner's entire input while still reporting that an
	adjudicator ran.

	rank(beam, tid, logprob) -> float, optional. Added to the pool key, so alignment evidence can
	reorder candidates without touching this function. None = pure LM ranking, which is the
	ablation baseline.

	adjudicator, optional. Lets alignment evidence, rather than the model's own distribution, order
	the candidates at an elapse position. Two methods, both given a live beam so per-lineage state
	can be looked up:

	  elapse_ids(beam, logprob_row) -> iterable of token ids to expand IN ADDITION to the model's
	      top-k. The row is passed because the useful set is the top-k among ELAPSE tokens, which
	      cannot be read off the overall top-k. Needed
	      because the top-k is not a superset of the choices worth considering: MEASURED on a width-4
	      dump, 130 of 399 elapse branch points had NO elapse token in the top-4 at all, and on the
	      case that motivated this the correct elapse sat at rank 3 of 4 with logprob -5.36 against
	      the argmax's -0.08. Harvesting from the top-k would leave the aligner nothing to vote on
	      exactly where its vote decides the rhythm.
	  losses(beam, cands, kind=...) -> a list parallel to `cands` of floats (lower is better) or
	      None. None means the aligner has no opinion about that candidate. If EVERY loss at a
	      position is None the pool falls back to pure log-probability order, so a run with an
	      adjudicator that cannot see anything is the same run as one without. `kind` is the branch
	      kind this row was gated on (BRANCH_ELAPSE or BRANCH_PITCH) and it selects WHICH quantity is
	      measured -- a tick forecast or a placed-note verdict. It is passed rather than re-derived
	      from `beam.state` so the value cannot drift from the one the gate used.

	The pool is then ordered on (loss, -logprob): alignment first, the model as the tie-break. It is
	deliberately not a weighted sum -- there is no lambda to fit, and the failure being corrected is
	the model being CONFIDENTLY wrong about a delta, which a finite weight against its own confidence
	cannot reliably outvote. Losses must be comparable ACROSS beams, since the pool is sorted whole;
	AlignState's decayed cost is (its bound is asserted in tests/midi/align_check.py).

	elapse_k: how many elapse tokens the adjudicator is asked for. 0 leaves the enumeration alone.

	observer(position, live, pool, kept, done) -> None, optional. Called once per decode position
	AFTER the cut, with the beams that were expanded, the whole sorted pool (kept and cut alike) and
	the survivors. A pool entry is (key, total, row, tid, align_loss), the last None where the
	adjudicator abstained or was absent. Read-only by contract: it exists so an inspector can record the search without
	being able to change it, which is what makes an inspected run and a plain run the same run. The
	cut candidates are the point -- a tree that shows only the survivors cannot answer why the
	search went the way it did.
	'''
	beam_size = max(1, int(beam_size))
	branch_k = max(1, int(branch_k))
	live = [Beam(state=(seed_state.clone() if seed_state is not None else BranchState()))]
	done = []
	# `branch_points` sums over LIVE BEAMS, so it is not a rate over positions -- with width B it can
	# reach B per position. `branch_positions` is the per-position count (any beam widened here), and
	# it is the one that answers "how often does the policy widen".
	rep = dict(branch_points=0, branch_positions=0, widened=0, expanded=0, steps=0, kinds={},
		# how often the aligner actually had an opinion, and how often it overturned the model. Both
		# are the numbers that say whether level 3 did anything, so they are counted rather than
		# asserted -- an adjudicator that silently abstains everywhere looks identical to none.
		adjudicated=0, abstained=0, overturned=0, forced_elapse=0, scored_only=0,
		# (position, beam) pairs at a real branch where the aligner was DELIBERATELY not consulted
		# because the lineage had not yet reached its first note_on AND an elapse token after it.
		# Distinct from `abstained`, which means the aligner WAS asked and had no evidence -- collapsing
		# the two would hide the gate, and the whole point of the elapse half is that it suppresses
		# branches that would otherwise have abstained anyway.
		pre_align=0,
		# candidates the logprob margin dropped, summed over (position, beam). Counted because the
		# margin is a DEFAULT: a run that prunes nothing and a run without the margin are the same run,
		# and only this number tells them apart.
		margin_pruned=0,
		# summed over (position, beam): how much of the vocabulary the automaton ruled out. 0 on a
		# run that never opened a mid-run branch, which is why it is counted rather than assumed.
		grammar_masked=0)
	forced = False
	ban_first = tuple(ban_first or ())
	# The vocabulary indexed the way the grammar mask below needs it: ids grouped by token class, and
	# for elapse tokens their tick values too (the automaton is per-value, not per-class). Built once
	# -- re-deriving the class of 582 tokens at every position of every beam is the same answer every
	# time, and the mask has to be cheap enough that nobody is tempted to make it optional.
	class_ids = {}
	elapse_table = []
	for tid, tok in enumerate(tokens):
		cls = token_class(tok, keywords)
		class_ids.setdefault(cls, []).append(tid)
		value = elapse_value(tok)
		if value is not None:
			elapse_table.append((tid, value))

	while live and len(live[0].ids) < max_new:
		first = not live[0].ids		# every live beam has the same length, so one test covers all
		logprobs = step([b.ids for b in live])
		rep['steps'] += 1
		if first and ban_first:
			logprobs = logprobs.clone()
			for row in range(logprobs.shape[0]):
				top = int(logprobs[row].argmax().item())
				if top in ban_first:
					forced = True
			for tid in ban_first:
				logprobs[:, tid] = float('-inf')

		pool = []
		widened_here = False
		vocab = int(logprobs.shape[-1])
		any_loss = False
		# The first position takes the model's argmax and nothing else: the aligner abstains there (no
		# pairs yet) so widening spends beam slots on continuations nothing can adjudicate. See
		# `greedy_first` in the docstring for the measurement.
		wide_here = beam_size > 1 and not (first and greedy_first)
		for row, beam in enumerate(live):
			kind = beam.state.branch_kind()
			width = branch_k if (kind != BRANCH_NONE and wide_here) else 1
			if width > 1:
				rep['branch_points'] += 1
				widened_here = True
				rep['kinds'][BRANCH_NAMES[kind]] = rep['kinds'].get(BRANCH_NAMES[kind], 0) + 1
			# GRAMMAR MASK. An elapse run is `[E1000]* [Exxx]? [Ex]?`, so at STAGE_MID only a LOW may
			# follow and at STAGE_LOW nothing may. Enforced on the DISTRIBUTION rather than left to
			# the branch kind, because `branch_kind` only decides whether extra candidates are
			# enumerated -- the model's own top-k reaches the pool either way, and so did the
			# adjudicator's forced elapse tokens. Unmasked, a --rank align run emitted `E160 E010` and
			# `E1 E1 E1 E1 E1 E1`: 10 of 23 runs malformed, every one of them overriding a LEGAL model
			# argmax at logprob ~-0.0000 with an illegal token at -5 to -21. The model is not the
			# problem here -- an LM-ranked run of the same checkpoint emitted 0 of 17 malformed.
			# A digit is masked for a SECOND reason: `[0-9a-f_]` is an argument digit, legal only after
			# the keyword whose argument it is, another digit of the same argument, or a channel
			# (`set_tempo 7 a 1 2 0`, `pitchwheel C4 2`). MEASURED over 8936 bare tokens in the corpus:
			# 6357 after a digit, 1730 after a keyword, 849 after a channel, 0 anywhere else. Unmasked
			# it reached the output as `E1a0 5 5 #53` -- digits belonging to no event.
			row_lp = logprobs[row]
			grammar = beam.state.grammar
			banned = [tid for tid, value in elapse_table if not grammar.admits(value)]
			for cls, ids in class_ids.items():
				if cls == CLS_ELAPSE:
					continue		# per-value, handled by the automaton above
				if not grammar.admits_class(cls):
					banned.extend(ids)
			if banned:
				row_lp = row_lp.clone()		# never in place: the other rows share this tensor
				for tid in banned:
					row_lp[tid] = float('-inf')
				rep['grammar_masked'] += len(banned)
			# Is the aligner allowed to act on this beam at all? One test, read twice below -- once by
			# the forced-elapse enumeration and once by the ranking -- because the two must agree: a
			# candidate enumerated for the aligner's benefit and then ranked by the model is the worst
			# of both, and that is exactly what an ungated enumeration produced in the header.
			gated = align_from_first_elapse and not beam.state.saw_elapse_after_note_on
			k = min(width, vocab)
			top = row_lp.topk(k)
			if k > 1:
				rep['widened'] += 1
			cands = [(int(tid), lp)
				for lp, tid in zip(top.values.tolist(), top.indices.tolist())]
			# Against this ROW's own best, which is cands[0] -- topk returns sorted, so no max() is
			# needed and the argmax always survives with a margin of exactly 0. `-inf` entries (fewer
			# legal tokens than k after the grammar mask) fail the test and go too, which is the same
			# thing the old `math.isfinite` guard downstream did.
			if logprob_margin is not None and len(cands) > 1:
				floor = cands[0][1] - logprob_margin
				kept = [c for c in cands if c[1] >= floor]
				rep['margin_pruned'] += len(cands) - len(kept)
				cands = kept
			# The elapse tokens the aligner wants considered, whether or not the model ranked them.
			# Added only at a real elapse branch point: elsewhere an elapse is either illegal (after a
			# keyword) or not what is being decided, and forcing one in would search a position the
			# grammar has already settled.
			# `wide_here`, not `beam_size > 1`: a forced elapse proposal is a widening like any other,
			# and at the first position it would reopen exactly what greedy_first just closed.
			#
			# Appended AFTER the logprob margin and deliberately exempt from it: a proposal exists
			# because the model ranked the token badly (the dump's cheapest real elapse sat 8.97 nats
			# behind the argmax), so a margin over these would delete the aligner's whole input and
			# quietly turn an adjudicated run back into an LM-only one.
			#
			# And gated on the same latch as the ranking below, for the same reason and then one more.
			# The reason: these candidates exist ONLY to give the aligner something to rank, so
			# enumerating them where the aligner is not allowed to rank spends beam slots on tokens
			# nothing will adjudicate -- the model already declined to rank them itself. The one more:
			# in the header they are actively harmful. MEASURED on the committed dump, 21 branch points
			# before the gate opened drew forced elapse candidates into a pool the LM then ordered,
			# so header positions competed against elapse tokens the model rated 14+ nats worse, and
			# the surviving lineages carried that spend forward to positions where evidence did exist.
			if (adjudicator is not None and elapse_k and wide_here
					and kind == BRANCH_ELAPSE and not gated):
				seen = {tid for tid, _ in cands}
				extra = 0
				for tid in adjudicator.elapse_ids(beam, row_lp):
					tid = int(tid)
					if tid in seen or not (0 <= tid < vocab):
						continue
					# The mask is not advice. An adjudicator proposes on rhythm evidence and has no
					# view on the automaton, so without this a forced token walked straight past it.
					if row_lp[tid].item() == float('-inf'):
						continue
					seen.add(tid)
					cands.append((tid, float(logprobs[row][tid].item())))
					extra += 1
				rep['forced_elapse'] += extra
			losses = None
			# At a real BRANCH point only -- elapse or pitch -- and never elsewhere. Ungated, the
			# epsilon rule reorders positions where the grammar owes an argument that the aligner has
			# no view on: measured on a 93-position run, 10 of 16 overturns were an argument token
			# (#4c, a digit) displaced by another lineage's token on a loss difference of ~1e-4, which
			# is not a rhythm verdict at all. A pitch branch is not in that category -- the pitch IS
			# the thing the alignment judges, and `observe` scores every one of them including a miss.
			#
			# The two kinds are scored on DIFFERENT quantities (a forecast against the next source
			# onsets in tick space, versus a verdict on a note actually placed) and land in the SAME
			# whole-pool sort, because both are `AlignState.cost * CostStepAttenuation + <bounded
			# term>` -- one recursion, one scale. See AlignState.forecast's docstring, which makes the
			# claim explicitly, and align_check.py, which asserts the shared bound.
			#
			# And not before this lineage's first note_on, nor before the first elapse token after it.
			# Until the note_on the file is still in its header, where the alignment has nothing to say
			# and the language model is nearly certain: ranking there let `Eb40` at logprob -19.11 beat
			# `ticks_per_beat` at -0.00000 and skip the header entirely -- which scored WELL, because
			# compose_output then supplies the source's header as a fallback, so the alignment was
			# buying good numbers with a decision that is not its to make.
			#
			# And until that first elapse the aligner's answer here is not a judgement but a None:
			# `forecast` needs a tick ratio, and `_update_ratio` cannot form one while every target tick
			# is still 0 (`slope = d_tgt / d_src` sits behind `if slope > 0`). MEASURED: all 47 elapse
			# branches on the committed dump's winning chain abstained. Both are latches on BranchState
			# and seed_state walks the primer through `feed`, so this restricts the piece's OPENING and
			# not every window's opening.
			if adjudicator is not None and kind in (BRANCH_ELAPSE, BRANCH_PITCH) and not gated:
				losses = adjudicator.losses(beam, cands, kind=kind)
				if any(x is not None for x in losses):
					any_loss = True
			elif adjudicator is not None and kind in (BRANCH_ELAPSE, BRANCH_PITCH) and gated:
				rep['pre_align'] += 1
			for i, (tid, lp) in enumerate(cands):
				total = beam.logprob + lp
				key = total + (rank(beam, tid, lp) if rank is not None else 0.0)
				pool.append((key, total, row, tid, None if losses is None else losses[i]))
		rep['expanded'] += len(pool)
		rep['branch_positions'] += widened_here

		# Sort the WHOLE pool, not each beam's own candidates: a beam that would have won on its
		# second-choice token has to be able to lose to another beam's first choice, which is the
		# only thing that makes this a search rather than N independent greedy runs. Ties break on
		# (row, tid) so a run is reproducible and beam_size=1 does not depend on the sort being
		# stable across float equality.
		lm_order = sorted(pool, key=lambda item: (-item[0], item[2], item[3]))
		if any_loss and not adjudicate:
			# SCORE-ONLY: the adjudicator was asked and its verdicts are recorded for the observer, but
			# the search still ranks on the model. This is what makes a dump diagnostic rather than a
			# different run -- the tree is the one the model actually builds, annotated with what the
			# aligner thought of each candidate in it. Ranking on the loss changes which nodes exist,
			# so the two questions ("what does the aligner think of this tree" and "what tree does the
			# aligner build") cannot be answered by one run.
			rep['scored_only'] += 1
			pool = lm_order
		elif any_loss:
			# Alignment first, the model as the tie-break. A candidate the aligner abstained on sorts
			# after every one it scored rather than at an invented value: `losses` fills the abstained
			# entries whose neighbours it can reach (that is the caller's job), so a None surviving to
			# here means no candidate at this position had a verdict to inherit from.
			rep['adjudicated'] += 1
			pool.sort(key=lambda item: (item[4] is None, item[4] if item[4] is not None else 0.0,
				-item[0], item[2], item[3]))
			if pool[0][:4] != lm_order[0][:4]:
				rep['overturned'] += 1
		else:
			if adjudicator is not None:
				rep['abstained'] += 1
			pool = lm_order

		nxt = []
		for _key, total, row, tid, _loss in pool:
			if len(nxt) >= beam_size:
				break
			parent = live[row]
			if tid == eos_id:
				# The terminator is not kept: it ends this window, and the caller's output stream is
				# one continuous piece, so a mid-stream terminator would be a stray token.
				fin = parent.clone(uid=parent.uid)
				fin.finished = True
				fin.logprob = total
				done.append(fin)
				continue
			child = parent.clone()
			child.ids.append(tid)
			child.logprob = total
			# An id past the vocab degrades to '<unknown>' rather than raising: a vocab/checkpoint
			# mismatch should not crash a run mid-file. The cost is that '<unknown>' classes as
			# CLS_SPECIAL, after which an elapse token is legal, so a stream full of them branches at
			# EVERY position -- an observed run on a checkpoint whose lm_head was 838 wide against a
			# 582-token vocab reported a branch rate of 1.000 for exactly this reason. Read a rate at
			# or near 1.000 as a vocab mismatch to go fix, not as the branch policy's behaviour.
			child.state.feed(tokens[tid] if 0 <= tid < len(tokens) else '<unknown>', keywords)
			nxt.append(child)
		if observer is not None:
			observer(len(live[0].ids) if live else 0, live, pool, nxt, done)
		live = nxt
		if len(done) >= beam_size:
			# Enough finished hypotheses that no live one can be needed: each live beam already
			# scores below the cut that produced these, and extending it only lowers its logprob
			# further (log p <= 0 per token). With length_alpha > 0 a longer beam CAN out-score a
			# shorter one, which is why this waits until the finished set is beam_size deep.
			break

	# A live beam that hit max_new never saw the terminator; it is still a legitimate answer for this
	# window (the greedy path returns exactly that), so it competes on the same score.
	pool = done + live
	rep['considered'] = len(pool)
	rep['finished'] = len(done)
	# Who won, decided by the same expression that picks the return value below -- not re-derived by a
	# caller. length_alpha can decide this when finished hypotheses of different length are in play,
	# so a caller guessing "the top kept candidate" would name the wrong lineage exactly when the
	# normalisation mattered.
	if pool:
		rep['best_uid'] = max(pool, key=lambda b: b.score(length_alpha)).uid
	if report is not None:
		for key, value in rep.items():
			if key == 'kinds':
				for k, v in value.items():
					report.setdefault('kinds', {})
					report['kinds'][k] = report['kinds'].get(k, 0) + v
			elif key == 'best_uid':
				report[key] = value			# an identity, not a count: summing it would be nonsense
			else:
				report[key] = report.get(key, 0) + value
	if not pool:
		return [], forced, rep
	best = max(pool, key=lambda b: b.score(length_alpha))
	return best.ids, forced, rep


def branch_profile (ids, tokens, keywords, seed_state=None):
	'''Count branch points a stream WOULD have had, per kind. For sizing a run before paying for it.

	Reads a finished token sequence, so it costs nothing and needs no model. The point is that the
	branch policy's cost is knowable in advance: given a greedy output, this says how many positions a
	beam would have widened at and therefore what the wall-clock multiplier will be.
	'''
	state = seed_state.clone() if seed_state is not None else BranchState()
	counts = {name: 0 for name in BRANCH_NAMES.values()}
	for tid in ids:
		counts[BRANCH_NAMES[state.branch_kind()]] += 1
		state.feed(tokens[tid] if 0 <= tid < len(tokens) else '<unknown>', keywords)
	total = max(1, len(ids))
	return dict(counts, positions=len(ids),
		branch_rate=(counts['elapse'] + counts['pitch']) / total)
