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

from .align import GrammarState, CLS_ELAPSE, CLS_KEYWORD, CLS_SPECIAL


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

	__slots__ = ('grammar', 'pitch_pending', 'velocity_pending')

	def __init__ (self, grammar=None, pitch_pending=False, velocity_pending=False):
		self.grammar = grammar if grammar is not None else GrammarState()
		# True between a note_on/note_off keyword and its `#XX`: exactly the positions where a pitch
		# token is the expected next thing.
		self.pitch_pending = pitch_pending
		# True between a note_on's pitch and its velocity, where nothing is being decided.
		self.velocity_pending = velocity_pending

	def clone (self):
		return BranchState(self.grammar.clone(), self.pitch_pending, self.velocity_pending)

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
		return BRANCH_ELAPSE if self.grammar.elapse_allowed else BRANCH_NONE

	def feed (self, tok, keywords):
		'''Commit one token string. Returns its grammar class.'''
		open_kw = self.grammar.open_keyword		# read BEFORE feed clears it
		cls = self.grammar.feed(tok, keywords)
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
	adjudicator=None, elapse_k=0):
	'''Run the search. Returns (best ids, forced, report).

	step(rows) -> log-probability rows. `rows` is a list of generated-id lists, one per live beam,
	ALL THE SAME LENGTH (a finished beam leaves the pool, so the batch stays rectangular and needs no
	padding -- and therefore no attention mask, which is one fewer thing that can be silently
	wrong). It must return a [len(rows), vocab] float tensor of log-probabilities.

	ban_first: token ids removed from the FIRST generated position. The caller's terminator ban lives
	here. `forced` reports whether the argmax at that position was banned, because the caller keys
	its end-of-piece detection on that having happened -- a ban that hid itself would leave the
	caller unable to tell a finished piece from a stalled one.

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
	  losses(beam, cands) -> a list parallel to `cands` of floats (lower is better) or None. None
	      means the aligner has no opinion about that candidate. If EVERY loss at a position is None
	      the pool falls back to pure log-probability order, so a run with an adjudicator that cannot
	      see anything is the same run as one without.

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
		adjudicated=0, abstained=0, overturned=0, forced_elapse=0)
	forced = False
	ban_first = tuple(ban_first or ())

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
		for row, beam in enumerate(live):
			kind = beam.state.branch_kind()
			width = branch_k if (kind != BRANCH_NONE and beam_size > 1) else 1
			if width > 1:
				rep['branch_points'] += 1
				widened_here = True
				rep['kinds'][BRANCH_NAMES[kind]] = rep['kinds'].get(BRANCH_NAMES[kind], 0) + 1
			k = min(width, vocab)
			top = logprobs[row].topk(k)
			if k > 1:
				rep['widened'] += 1
			cands = [(int(tid), lp)
				for lp, tid in zip(top.values.tolist(), top.indices.tolist())]
			# The elapse tokens the aligner wants considered, whether or not the model ranked them.
			# Added only at a real elapse branch point: elsewhere an elapse is either illegal (after a
			# keyword) or not what is being decided, and forcing one in would search a position the
			# grammar has already settled.
			if (adjudicator is not None and elapse_k and beam_size > 1
					and kind == BRANCH_ELAPSE):
				seen = {tid for tid, _ in cands}
				extra = 0
				for tid in adjudicator.elapse_ids(beam, logprobs[row]):
					tid = int(tid)
					if tid in seen or not (0 <= tid < vocab):
						continue
					seen.add(tid)
					cands.append((tid, float(logprobs[row][tid].item())))
					extra += 1
				rep['forced_elapse'] += extra
			losses = None
			if adjudicator is not None:
				losses = adjudicator.losses(beam, cands)
				if any(x is not None for x in losses):
					any_loss = True
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
		if any_loss:
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
