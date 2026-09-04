'''Beam-search translation of a midiseq2 file, over the same sliding window as translateMidiseq2.py.

A SEPARATE script on purpose. translateMidiseq2.py is the measured, working greedy translator whose
numbers the whole tuning record is stated in (src_window 960 / prime_window 320 -> onsetF1 0.514,
pitchF1 0.878 on 20260814-l16d256 e358); this adds a search on top and the two have different
reasons to change. It IMPORTS that script rather than copying it, so the window logic, the source
cursor, the primer trimming and the output stitching are literally the same code -- a beam run and a
greedy run can only differ in which token each step chose.

The subclasses override exactly one method, `generate`. Everything else -- source_window,
build_prefix, advance_output, trim_prime, finished, advance_source_by_onsets, translate -- is
inherited untouched, which is what makes `--beam 1` a real parity check rather than a similar
pipeline that happens to agree.

  BRANCH POINTS  elapse and pitch only (starry/midi/beam.py). Elsewhere the beam takes its argmax.
                 Measured 86.2% of positions still branch -- an elapse token is legal nearly
                 everywhere -- so beam_size, not the policy, is what bounds the cost: 3.7x greedy
                 wall-clock at width 4 on CPU.
  SEAM           the beam collapses to one hypothesis per window. Carrying B beams across the seam
                 would fork the sliding state itself: each beam rolls out a different number of
                 measures, so it would own its own prime_start, its own source cursor (advanced by
                 counting note_ons in what IT rolled out) and its own output. That is B independent
                 translations whose cost grows with the file, not a beam search.
  DETERMINISTIC  no temperature. Beam search and sampling answer different questions, and a sampled
                 beam is neither the model's argmax path nor a draw from its distribution.

Ablation ladder (1 = greedy, already the other script):
  2. beam, LM-only ranking            --beam 4 --rank lm
  3. beam + alignment ranking         <- HERE. --beam 4 --rank align. The alignment FORECASTS the tick
                                      an elapse candidate would fix and orders the pool on it, the
                                      model breaking ties; changes no token's legality. Level 3 with
                                      an alignment that abstains everywhere is exactly level 2, which
                                      is asserted rather than assumed.
  4. beam + alignment mask            not yet wired: prunes infeasible elapse tokens
Evaluate with tests/midi/translate_accuracy_check.py --skip-bars 6 --score-bars, PER FILE and paired
against a greedy run of the same checkpoint. Never on means: the per-file spread on this task is
larger than the effect (0.345 vs 0.018), so a mean can and has pointed the opposite way.

Run:
  python tools/midi/translateMidiseq2Beam.py --run <run_dir> --input a.midiseq2.txt --beam 4
  python tools/midi/translateMidiseq2Beam.py --run <run_dir> --input a.txt --beam 1   # == greedy
'''

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from starry.midi.beam import (BranchState, beam_search, BRANCH_NAMES, BRANCH_ELAPSE, BRANCH_PITCH)
from starry.midi.align import (AlignState, Config, STAGE_LOW, elapse_class, elapse_value,
	soft_delta, soft_indices)

# The greedy translator is the base, not a template: these are the same objects it uses.
from translateMidiseq2 import (DEFAULT_RUN, SlidingTranslator, SlidingEncDecTranslator,
	resolve_checkpoint, resolve_tokenizer, load_model, render_lines, compose_output, write_output,
	report_output, source_header, encode_lines, note_on_events, keyword_tokens)
from starry.utils.config import Configuration


class BeamMixin:
	'''Adds beam search to a SlidingTranslator by overriding `generate` and nothing else.

	`translate` calls `self.generate(...)`, so a subclass is the whole integration: no sliding-window
	code is duplicated, re-derived, or forked. beam_size=1 falls through to the parent's greedy
	implementation by an explicit early return, so a width-1 run does not merely resemble greedy --
	it IS greedy, running the same lines.
	'''

	def __init__ (self, *args, beam_size=1, branch_k=4, length_alpha=0.7, inspector=None,
		position_cap=0, adjudicator=None, elapse_k=0, adjudicate=True, greedy_first=True,
		align_from_first_elapse=True, logprob_margin=2.0, **kwargs):
		super().__init__(*args, **kwargs)
		self.beam_size = max(1, int(beam_size))
		self.branch_k = max(1, int(branch_k))
		self.length_alpha = length_alpha
		self.beam_report = {}		# accumulated across steps; read after translate() returns
		self.inspector = inspector
		self.adjudicator = adjudicator
		# Whether the adjudicator's verdicts ORDER the pool, as opposed to merely being recorded. An
		# inspection run wants them recorded unconditionally -- the whole point of a dump is to ask
		# what the aligner thought of every candidate -- but ranking on them builds a DIFFERENT tree,
		# so the two are separate decisions and only --rank align makes the second one.
		self.adjudicate = adjudicate
		self.elapse_k = max(0, int(elapse_k))
		# Take the model's argmax at each window's first generated position and do not branch there.
		# The aligner has no pairs at that position and abstains outright, so the extra slots go to
		# continuations nothing can adjudicate: MEASURED at position 0 of the align dump, the argmax was
		# `ticks_per_beat` at logprob -0.00000 and the other three slots went to tokens the model rated
		# 16.77, 18.05 and 19.11 nats worse, all with loss None.
		self.greedy_first = bool(greedy_first)
		# Hold the aligner off until the piece has a note_on AND an elapse token after it. The second
		# half is not cosmetic: until a tick advances, `_update_ratio` cannot form a ratio and every
		# forecast is None, so the branches before it were being asked a question with no answer.
		self.align_from_first_elapse = bool(align_from_first_elapse)
		# Drop a candidate rated more than this many nats below its own row's best. `branch_k` is a
		# fixed width, so without it the search spends the full width wherever the model is certain --
		# MEASURED at the dump's first post-header elapse branch, all four rows put ~-0.0000 on
		# elapse-zero with the next candidate 8.97 nats back. None disables it.
		self.logprob_margin = logprob_margin
		# --inspect N caps the tokens a window generates, not just the tokens recorded: generating
		# 2000 and keeping 40 would spend the whole run to dump a fragment of its first window.
		self.position_cap = max(0, int(position_cap))
		self.step_index = 0

	def seed_state (self, prefix_ids, n_source=None):
		'''Grammar state at the first generated position, walked over the TARGET half of the prefix.

		Seeded rather than empty because the first generated token continues whatever the primer left
		open -- a half-emitted event, or an open elapse run. A fresh state would call that position a
		boundary and would put branch points in the wrong places.

		Derived from the prefix rather than taken as an argument: the inherited `translate` does not
		pass the primer to `generate`, and adding a parameter to it would mean editing the greedy
		script, which is the thing this file exists to avoid. The prefix already carries the answer --
		it is `source ++ <sep> ++ target` -- so the target half is prefix[n_source:]. `<sep>` and
		`<bos>` walk through the grammar as CLS_SPECIAL and close any open event, which is exactly
		what they mean, so they need no special case.
		'''
		state = BranchState()
		vocab = self.tk.tokens
		if n_source is None:
			# Fall back to locating <sep> rather than guessing: a wrong split would seed the state
			# from source tokens and silently move every branch point in the step.
			try:
				n_source = prefix_ids.index(self.tk.sep_id)
			except ValueError:
				return state
		for tid in prefix_ids[n_source:]:
			state.feed(vocab[tid] if 0 <= tid < len(vocab) else '<unknown>', self.keywords)
		return state

	def rank_fn (self):
		'''Extra per-candidate score added to the pool key. None = pure LM (ablation level 2).

		A hook, deliberately empty at this level: the alignment ranking is level 3 and lands here
		without touching the search or the window logic.
		'''
		return None

	@torch.no_grad()
	def generate (self, prefix_ids, prefix_positions, temperature, top_k, top_p, n_source=None,
		next_position=None):
		if self.beam_size <= 1:
			# Not "equivalent to" greedy: the same code path, so parity is structural.
			return super().generate(prefix_ids, prefix_positions, temperature, top_k, top_p,
				n_source=n_source, next_position=next_position)
		if temperature:
			raise ValueError(f'beam search is deterministic; got temperature {temperature}')

		ctx = self.step_begin(prefix_ids, prefix_positions, n_source=n_source)
		base_pos = prefix_positions[-1] if prefix_positions else 0

		def positions_for_row (n):
			'''RoPE positions for n generated tokens, matching the greedy loop exactly.

			The greedy path appends `next_position` for the FIRST generated token when pos_style is
			absolute and prev+1 thereafter; reproducing that here rather than approximating it is the
			difference between a beam that searches the same space and one that searches a shifted
			one.
			'''
			out = []
			prev = base_pos
			for i in range(n):
				prev = (next_position if i == 0 and next_position is not None else prev + 1)
				out.append(prev)
			return out

		def step (rows):
			logits = self.step_logits(ctx, rows, positions_for_row)
			return F.log_softmax(logits.float(), dim=-1)

		# The greedy loop's condition is `while len(ids) < max_token` over prefix ++ generated, so the
		# room is exactly this. NOT max(1, room): a floor of one token would generate where greedy
		# generates nothing, which is a parity break rather than a safety net. translate() already
		# refuses to call generate when the prefix has filled the budget.
		room = self.max_token - len(prefix_ids)
		if self.position_cap:
			room = min(room, self.position_cap)
		observer = None
		if self.inspector is not None:
			self.inspector.begin_window(self.step_index, n_source,
				list(prefix_ids[n_source:]) if n_source else [])
			observer = self.inspector.observe
		ids, forced, _rep = beam_search(step, self.tk.tokens, self.keywords, self.tk.eos_id,
			max_new=room, beam_size=self.beam_size, branch_k=self.branch_k,
			length_alpha=self.length_alpha, observer=observer,
			adjudicator=self.adjudicator, elapse_k=self.elapse_k, adjudicate=self.adjudicate,
			seed_state=self.seed_state(list(prefix_ids), n_source),
			# Same first-position terminator ban as the greedy path, for the same measured reason: an
			# <eos> logit of 13.16 against 7.71 for next-best is not something a finite penalty can be
			# tuned against. `forced` must keep flowing back, because translate() keys its
			# end-of-piece detection on it.
			ban_first=(self.tk.eos_id,), greedy_first=self.greedy_first,
			align_from_first_elapse=self.align_from_first_elapse,
			logprob_margin=self.logprob_margin,
			rank=self.rank_fn(), report=self.beam_report)
		if self.inspector is not None:
			# Which hypothesis won, taken from the search's own report rather than re-derived here, so
			# the next window's alignment continues from the lineage the output actually took.
			self.inspector.end_window(self.beam_report.get('best_uid'))
		self.step_index += 1
		return ids, forced


class BeamTranslator (BeamMixin, SlidingTranslator):
	'''Decoder-only: one growing prefix, batched across beams in the batch dim.'''

	def step_begin (self, prefix_ids, prefix_positions, n_source=None):
		return dict(ids=list(prefix_ids), positions=list(prefix_positions))

	@torch.no_grad()
	def step_logits (self, ctx, rows, positions_for_row):
		'''Next-position logits for every live beam, in ONE forward pass.

		The rows are all the same length by construction (a finished beam leaves the pool), so the
		batch is rectangular and needs no padding -- and therefore no attention mask, which is one
		fewer thing that can be silently wrong. The assertion keeps that a fact rather than an
		assumption.
		'''
		widths = {len(r) for r in rows}
		assert len(widths) == 1, f'ragged beam batch {sorted(widths)}'
		n = len(rows[0])
		tail = positions_for_row(n)
		ids = torch.tensor([ctx['ids'] + row for row in rows], dtype=torch.long, device=self.device)
		pos = torch.tensor([ctx['positions'] + tail for _ in rows], dtype=torch.long,
			device=self.device)
		return self.model(ids, None, pos)[:, -1, :]


class BeamEncDecTranslator (BeamMixin, SlidingEncDecTranslator):
	'''EncDec: the source is encoded ONCE and its memory shared across beams.

	The encoder pass does not depend on the hypothesis, so a width-B search costs one encode plus B
	decoder columns, not B encodes.
	'''

	def step_begin (self, prefix_ids, prefix_positions, n_source=None):
		if n_source is None:
			raise ValueError('EncDec generation requires the source prefix length')
		if n_source <= 0 or n_source >= len(prefix_ids):
			raise ValueError(f'invalid EncDec prefix split {n_source}/{len(prefix_ids)}')
		source_ids = torch.tensor([prefix_ids[:n_source]], dtype=torch.long, device=self.device)
		source_pos = torch.tensor([prefix_positions[:n_source]], dtype=torch.long, device=self.device)
		masks = torch.ones_like(source_ids)
		memory = self.model.encode(source_ids, masks, source_pos)
		return dict(memory=memory, masks=masks, ids=list(prefix_ids[n_source:]),
			positions=list(prefix_positions[n_source:]))

	@torch.no_grad()
	def step_logits (self, ctx, rows, positions_for_row):
		widths = {len(r) for r in rows}
		assert len(widths) == 1, f'ragged beam batch {sorted(widths)}'
		n = len(rows)
		tail = positions_for_row(len(rows[0]))
		ids = torch.tensor([ctx['ids'] + row for row in rows], dtype=torch.long, device=self.device)
		pos = torch.tensor([ctx['positions'] + tail for _ in rows], dtype=torch.long,
			device=self.device)
		# Keep the newest decoder context but preserve its original RoPE positions (as the parent does).
		ids = ids[:, -self.model.max_seq_len:]
		pos = pos[:, -self.model.max_seq_len:]
		# expand, not repeat: the memory is read-only here, so B rows can share one allocation
		memory = ctx['memory'].expand(n, *ctx['memory'].shape[1:])
		masks = ctx['masks'].expand(n, ctx['masks'].shape[1])
		return self.model.decode(memory, ids, masks, pos)[:, -1, :]


class LineageTracker:
	'''Per-hypothesis tick cursor, note_on walk state and AlignState, keyed on Beam.uid.

	One table, shared by the adjudicator (which READS a parent's state to forecast a candidate,
	before the cut) and the inspector (which ADVANCES it into the surviving children, after the cut).
	They are deliberately not given a table each: two clocks over the same token stream that are
	never compared is how a tick counter drifts away from the notes it is counting, and align.py
	makes the same argument about GrammarState's accumulator versus note_on_events' absolute tick.

	The ordering the search gives us is what makes the sharing safe -- enumerate (adjudicator reads
	parents) -> sort -> cut -> observer (inspector writes children) -- so a parent's entry is still
	present when the adjudicator asks for it, and pruning happens only after the children exist.

	AlignState.clone() is what makes this affordable: a candidate that will be cut is scored on a
	clone that is then dropped, so scoring the road not taken costs nothing permanent.
	'''

	def __init__ (self, tokenizer, keywords, src_events, seed_offset=0.0):
		self.tk = tokenizer
		self.keywords = keywords
		self.src_events = src_events
		self.seed_offset = seed_offset
		self.table = {}					# uid -> lineage dict, pruned to the live set each position
		self.root = self.fresh()

	def fresh (self):
		return dict(align=AlignState(self.src_events, seed_offset=self.seed_offset),
			walk=None, tick=0, prev_onset=None, softindex=0.0, si=0.0)

	def base (self, uid):
		'''The lineage state for `uid`, defaulting to the carried root at a window's first position.'''
		return self.table.setdefault(uid, self.root)

	def advance (self, base, tid, commit=False):
		'''Walk ONE token on top of `base` -> (new lineage dict, alignment verdict or None).

		The verdict is None unless this token CLOSED a note_on: an elapse moves the clock but has
		nothing to match yet, and a velocity or channel is not a musical event at all. So a node
		carries an alignment score exactly when it created something the alignment can judge.

		ALWAYS clones, and `commit` is now only about whether the caller stores the result. Advancing
		the parent's AlignState in place was wrong: one parent can contribute SEVERAL survivors -- it
		does at 11.5% of positions on the committed dump, up to 4 at once -- and each of them would
		then fold its own note into the same object, so the second sibling would be scored against a
		state already carrying the first sibling's note. MEASURED on that dump: 7 positions had one
		parent with two or three surviving children that each closed a note_on, so every align figure
		recorded after the first of them is against a polluted state. The adjudicator makes this worse
		than a recording defect, since it FORECASTS from `base['align']`.

		`commit` is kept in the signature because the recording still distinguishes a candidate that
		became a lineage from one that was scored and dropped, and because the flag documents intent
		at the call site.
		'''
		align = base['align'].clone()
		events, tick, walk = note_on_events([tid], self.tk, self.keywords,
			tick0=base['tick'], state=base['walk'])
		prev_onset, softindex = base['prev_onset'], base['softindex']
		detail = None
		for e in events:
			# softIndex is a sum of tanh steps over intervals, so it extends incrementally: the whole
			# onset list is never needed, which is exactly what lets a PARTIAL hypothesis be scored.
			if prev_onset is not None:
				softindex += soft_delta(e['onset'] - prev_onset)
			prev_onset = e['onset']
			detail = dict(align.observe(e['pitch'], e['onset'], softindex))
			detail.update(pitch=e['pitch'], onset=e['onset'], softIndex=softindex)
			if detail.get('src') is not None:
				src = self.src_events[detail['src']]
				detail.update(src_onset=src['onset'], src_pitch=src['pitch'])
		return dict(align=align, walk=walk, tick=tick, prev_onset=prev_onset,
			softindex=softindex, si=softindex if prev_onset is None
				else softindex + soft_delta(tick - prev_onset)), detail

	def keep (self, uid, state):
		self.table[uid] = state

	def prune (self, alive):
		'''Drop lineages nothing points at, so the table stays proportional to the beam width.'''
		self.table = {u: v for u, v in self.table.items() if u in alive}

	def carry (self, best_uid):
		'''Adopt the winning lineage as the root the next window continues from.'''
		self.root = self.table.get(best_uid, self.root)


class AlignAdjudicator:
	'''Lets the alignment, not the model's own distribution, order the candidates at a branch point.

	BOTH branch kinds, and they are asymmetric. At an ELAPSE point nothing has been placed, so the
	candidates are scored by forecast; at a PITCH point the tick is already settled and `observe`
	applies directly, so they are scored by verdict. The elapse case came first and the rest of this
	docstring is about it; the pitch case is in `_pitch_losses`, and the short version is that leaving
	the pitch to the language model meant the note itself -- the one thing the alignment exists to
	judge -- was the only decision the aligner never saw.

	The defect this addresses, measured on the committed width-4 dump: of 554 elapse candidates, 0
	carried an alignment verdict, because AlignState only observes a note_on and the pitch that would
	let it score arrives one or two tokens after the elapse that fixed the rhythm. So the decision
	that determines the timing was taken on the language model alone -- and the model's elapse
	distribution is high-entropy (top-4 spanning 1.30 to 2.50 nats on the motivating case), which
	makes the margins there noise rather than judgement. On that case the correct elapse lost by 0.15
	nats while the aligner ranks it first by a clear margin.

	Three things have to be true for the aligner's vote to reach the decision, and each was a hole:

	  ENUMERATION  the model's top-k is not a superset of the choices worth considering. 130 of 399
	               elapse branch points had no elapse token in the top-4 at all, and on the motivating
	               case the correct one sat at rank 3 with logprob -5.36 against the argmax's -0.08.
	               So the elapse candidates are enumerated by taking the top `elapse_k` AMONG elapse
	               tokens, independently of where they fall in the overall distribution.
	  SCORING      a tick has no pitch yet, so `observe` cannot be used. AlignState.forecast scores a
	               tick against the source's next unmatched onsets in TICK space (see its docstring
	               for why softIndex cannot work here: soft_delta saturates and the estimate goes flat
	               across exactly the choices needing separation).
	  COVERAGE     not every candidate at an elapse point fixes an onset. note_off, a controller and a
	               velocity produce no onset, so the aligner has genuinely nothing to say about them,
	               and inventing a number would be worse than the honest gap. They inherit instead --
	               see `_inherit`. This is the one hole a PITCH point does not have: every pitch
	               candidate is measurable, misses included.

	A note_on is scored as ELAPSE ZERO, which is the substance of treating this as one decision: at a
	position where an elapse is legal, emitting a type token instead is a choice that the next event
	happens NOW, and it belongs on the same axis as the choice that it happens 480 ticks from now.
	Ranking them separately is what let a nearly-free syntactic continuation outrank every real
	rhythm option.
	'''

	# Separation given to a candidate that inherits its loss from a neighbour in the language model's
	# order. Small enough not to reorder anything the aligner actually scored (forecast differences on
	# real material are 1e-1), large enough to survive float addition and keep the inherited group in
	# the model's own order rather than collapsing it into a tie.
	EPSILON = 1e-4

	def __init__ (self, tracker, tokenizer, elapse_k=8):
		self.tracker = tracker
		self.tk = tokenizer
		self.elapse_k = max(1, int(elapse_k))
		# token id -> elapse tick value, for every elapse token in the vocabulary. Built once: the
		# alternative is parsing a token string per candidate per beam per position.
		self.elapse_values = {}
		for tid, tok in enumerate(tokenizer.tokens):
			value = elapse_value(tok)
			if value is not None:
				self.elapse_values[tid] = value
		# What `elapse_ids` may PROPOSE from: every elapse token EXCEPT the 15 lows (0x1..0xf).
		# `elapse_values` above stays complete, because a low still reaches the pool whenever the
		# model's own top-k contains one and must remain scorable -- this set only bounds what the
		# aligner REINFORCES.
		#
		# A low advances the tick by 1..15, which against a 480-tick beat is a rounding correction and
		# not a rhythmic choice, so a proposal slot spent on one buys no option worth a beam slot.
		# MEASURED on the pre-change 200-token dump: of 248 (position, beam) rows that drew proposals,
		# 29 spent a slot on a low while a MID was still legal, and 109 were at STAGE_MID where a low
		# is all the automaton admits. `BranchState.branch_kind` now returns BRANCH_NONE at STAGE_MID,
		# so those 109 rows are no longer branch points at all and never reach here -- which is why
		# this set needs no stage test of its own.
		self.proposal_ids = [tid for tid, value in self.elapse_values.items()
			if elapse_class(value) != STAGE_LOW]
		self.note_on_id = None
		for tid, tok in enumerate(tokenizer.tokens):
			if tok == 'note_on':
				self.note_on_id = tid
				break
		# The `#XX` tokens, for the pitch branch. A pitch is the ONLY candidate class the alignment can
		# judge directly -- `observe` needs a (pitch, tick) pair and the tick is already fixed by the
		# elapse run behind it -- so this set is exactly the set that gets a measured loss there.
		self.pitch_ids = {tid for tid, tok in enumerate(tokenizer.tokens)
			if len(tok) > 1 and tok[0] == '#'}
		# Tokens that END a stream rather than place an event in it. Excluded from the inheritance pass
		# below; see the note there for the measurement that made this necessary.
		self.terminator_ids = {tid for tid, tok in enumerate(tokenizer.tokens)
			if tok in ('<eos>', '<eom>')}
		for attr in ('eos_id', 'eom_id'):
			tid = getattr(tokenizer, attr, None)
			if tid is not None:
				self.terminator_ids.add(int(tid))

	def elapse_ids (self, beam, logprob_row):
		'''The top `elapse_k` non-LOW elapse tokens by log-probability, plus note_on as the elapse-0 option.

		Top-k among ELAPSE tokens specifically, not among all tokens -- that is the whole point, since
		the overall top-k routinely contains no elapse at all. LOW tokens (0x1..0xf) are never
		proposed: see `proposal_ids`. A low still reaches the pool whenever the model's own top-k
		contains one, and is always scorable -- it is only never reinforced. Deliberately NOT masked by
		`feasible_elapse_tokens`: that mask needs a target INTERVAL, it is a separate ablation level,
		and folding it in here would make it impossible to tell which of the two changed a result.

		The GRAMMAR mask is not in that category and is not optional: `[E1000]* [Exxx]? [Ex]?` holds
		whatever the rhythm argues, and beam_search has already set the inadmissible ids to -inf in
		the row passed here, so proposing one is a no-op (it also drops them explicitly). Ranking on
		the masked row means a proposal is never spent on a token that cannot be taken.
		'''
		ranked = sorted(self.proposal_ids, key=lambda tid: -logprob_row[tid])
		out = ranked[:self.elapse_k]
		if self.note_on_id is not None:
			out.append(self.note_on_id)
		return out

	def losses (self, beam, cands, kind=BRANCH_ELAPSE):
		'''Align loss per candidate (lower better), or None where nothing could be inherited.

		Two passes, and the FIRST one is what `kind` selects -- the branch kind decides which quantity
		is measurable at this position:

		  BRANCH_ELAPSE  a FORECAST. Nothing has been placed yet, so each candidate is scored by the
		                 tick it would fix: an elapse token at `tick + value`, a note_on at `tick`
		                 (elapse zero). See `_elapse_losses`.
		  BRANCH_PITCH   a VERDICT. The tick is already fixed by the elapse run behind this position,
		                 so a pitch candidate is scored by actually placing the note -- which is the
		                 alignment's native question, not a forecast of it. See `_pitch_losses`.

		Both land in one whole-pool sort across beams (a position can have a pitch-branching beam and
		an elapse-branching beam at once -- measured, pos 140 of the committed dump had exactly that),
		which is sound because both are `AlignState.cost * CostStepAttenuation + <bounded term>`: one
		recursion, one scale, one bound.

		The SECOND pass is shared and fills in the rest by inheriting along the language model's own
		ordering -- see `_inherit`.
		'''
		out = (self._pitch_losses(beam, cands) if kind == BRANCH_PITCH
			else self._elapse_losses(beam, cands))
		return self._inherit(cands, out)

	def _elapse_losses (self, beam, cands):
		'''Forecast pass: the tick each candidate would fix, scored against the source's next onsets.

		COVERAGE is a real gap here and the honest shape for it is None: note_off, a controller and a
		velocity produce no onset, so the aligner has genuinely nothing to say about them. They go to
		`_inherit`.
		'''
		base = self.tracker.base(beam.uid)
		align, tick = base['align'], base['tick']
		out = [None] * len(cands)
		for i, (tid, _lp) in enumerate(cands):
			delta = self.elapse_values.get(tid)
			if delta is not None:
				out[i] = align.forecast(tick + delta)[0]
			elif tid == self.note_on_id:
				out[i] = align.forecast(tick)[0]			# elapse zero
		return out

	def _pitch_losses (self, beam, cands):
		'''Verdict pass: place each candidate pitch on the already-fixed tick and take its cost.

		This is the branch the search was missing. An elapse point had to be forecast because no note
		exists yet, but by the time a pitch is being chosen the tick is settled and `observe` -- the
		function the whole alignment is built around -- applies directly. Leaving it to the LM meant
		the note itself, the one thing the alignment exists to judge, was the one decision it never
		saw: MEASURED at pos 140 of the committed align dump, `#4a` had self_cost 4e-05 and cost
		0.13734 (the best at that position, against 0.8099 for the best SCORED candidate) AND a better
		logprob than the token that won, and was cut at rank 10 -- because `loss is None` is the first
		sort key, so an unmeasured candidate is not neutral, it is last.

		No EPSILON is needed for the misses. `observe` prices an unmatched note at
		`cost * CostStepAttenuation + MissCost`, on the same recursion as a match, so a candidate with
		no source counterpart sorts behind any decent match by construction rather than by a placement
		rule -- which settles the question of what to do with the `src=None` pitches (pos 140 had three
		of them at cost 1.13731 against the matched 0.13734).

		`tracker.advance` ALWAYS clones the AlignState, so scoring a candidate that will be cut leaves
		nothing behind. A pitch that somehow closes no event (a `#` under note_off, which cannot reach
		here because BRANCH_PITCH_EVENTS is note_on only) yields None and inherits.
		'''
		base = self.tracker.base(beam.uid)
		out = [None] * len(cands)
		for i, (tid, _lp) in enumerate(cands):
			if tid not in self.pitch_ids:
				continue
			_state, detail = self.tracker.advance(base, tid)
			if detail is not None:
				out[i] = detail['cost']
		return out

	def _inherit (self, cands, out):
		'''Fill the unmeasured entries by inheriting along the model's own order, +/- EPSILON per rank.

		The nearest preceding scored candidate's loss plus EPSILON, or, for the model's argmax which
		has nothing preceding it, the nearest following one's loss minus EPSILON. The epsilon
		accumulates with distance in that order, so a run of inheriting candidates keeps the model's
		ranking among themselves instead of collapsing to one value.

		This is a placement rule, not a measurement, and it is the honest shape for one: it says "this
		candidate is worth about what the rhythm choice next to it is worth, and the model
		prefers/disprefers it by a hair", which is exactly as much as is known about a token the
		aligner cannot see. Returning None for all of them instead would drop them behind every scored
		candidate, which would ban note_off at any position where an elapse was also legal.
		'''
		# A TERMINATOR never inherits. Inheriting is defensible for a token that stands in for a rhythm
		# choice -- note_off at elapse zero postpones the decision two positions and comes back to it --
		# but <eos>/<eom> end the stream and settle no onset, so anchoring one to a neighbour's rhythm
		# verdict prices the model's objection to it at EPSILON. Measured: at one position <eos> sat
		# 2e-4 behind the anchor while the model rated it 14.17 nats worse, which put it second in the
		# pool, into `done`, and four such terminations stopped the search at 93 of 200 positions with
		# one measure emitted. Left as None it sorts behind every scored candidate, which is the honest
		# placement: the aligner has nothing to say about ending the piece.
		blocked = {i for i, (tid, _lp) in enumerate(cands) if tid in self.terminator_ids}
		order = [i for i in sorted(range(len(cands)), key=lambda i: (-cands[i][1], i))
			if i not in blocked]
		scored = [pos for pos, i in enumerate(order) if out[i] is not None]
		if not scored:
			return out			# the aligner abstained everywhere; the search falls back to the LM
		for pos, i in enumerate(order):
			if out[i] is not None:
				continue
			before = [p for p in scored if p < pos]
			if before:
				src = before[-1]
				out[i] = out[order[src]] + self.EPSILON * (pos - src)
			else:
				src = scored[0]
				out[i] = out[order[src]] - self.EPSILON * (src - pos)
		return out


class BeamInspector:
	'''Records the search tree and what the alignment thinks of every candidate in it.

	Two questions an LM-only run cannot answer about itself: WHERE did the search have a real choice,
	and at those places, did the model's ranking agree with the alignment's? Both need the CUT
	candidates. A tree of survivors shows the path taken and nothing about the paths available, which
	is precisely the part under suspicion when a beam run scores worse than the greedy run it
	contains.

	Read-only with respect to the search. This is passed as beam_search's `observer`, which fires
	AFTER the cut, so nothing here can change which token was chosen: an inspected run and a plain
	run produce the same output. That is the only reason the recording is worth reading.

	Per-lineage state, keyed on Beam.uid: each hypothesis owns its tick cursor, its note_on walk
	state and its AlignState, because all three are functions of the tokens THAT hypothesis emitted.
	Sharing one would score every candidate against whichever beam happened to be walked last.
	AlignState.clone() is what makes it affordable -- a cut candidate is scored on a clone that is
	then dropped, so scoring the road not taken costs nothing permanent.
	'''

	def __init__ (self, tracker, limit=0, adjudicated=False):
		self.tracker = tracker
		# Whether an adjudicator ran AT ALL, which is not derivable from the losses: on an LM-only run
		# every one is None, and writing that as a `loss` key makes the viewer report 2393 candidates
		# as "abstained" when nothing was ever asked. Absent key = not adjudicated; present-and-null =
		# asked and had no evidence. Two different facts about the run.
		self.adjudicated = adjudicated
		self.tk = tracker.tk
		self.keywords = tracker.keywords
		self.src_events = tracker.src_events
		self.limit = limit				# 0 = uncapped; else stop recording after N positions per window
		self.windows = []
		self.window = None
		self.nodes = 0
		self.truncated = False

	def begin_window (self, step, n_source, prime_ids):
		'''Open a tree for one sliding window.

		The primer is NOT replayed into the alignment. It is a re-presentation of output the previous
		window already produced and already scored; folding it in again would count those notes
		twice and drag the running cost with it. The carried `root` is the winning lineage from the
		previous window, which is the state the new tokens actually continue.
		'''
		self.window = dict(step=step, n_source=n_source, best_uid=None,
			prime_tokens=[self.tk.tokens[t] for t in prime_ids
				if 0 <= t < len(self.tk.tokens)], positions=[])
		self.windows.append(self.window)

	def end_window (self, best_uid):
		# Recorded on the window, because the winning PATH is the tree's spine: without it a reader
		# cannot tell which of the kept branches the output actually came from, and every view that
		# greys out "what came after" has no baseline to grey out.
		if self.window is not None:
			self.window['best_uid'] = best_uid
		self.tracker.carry(best_uid)
		self.window = None

	def observe (self, position, live, pool, kept, done):
		if self.window is None:
			return
		if self.limit and len(self.window['positions']) >= self.limit:
			self.truncated = True
			return
		for beam in live:
			self.tracker.base(beam.uid)
		# Which pool entries became survivors, by the search's OWN rule: walk the sorted pool, and
		# the first len(kept) non-terminator entries are the ones it kept. Derived rather than
		# matched on token identity -- two beams can propose the same token at the same position, and
		# a match on (token, prefix) would then attach the recording to the wrong lineage.
		taken = 0
		cands = []
		for rank_i, (key, total, row, tid, loss) in enumerate(pool):
			parent = live[row]
			base = self.tracker.base(parent.uid)
			child = None
			if tid == self.tk.eos_id:
				pass				# went to `done`; it never becomes a live child
			elif taken < len(kept):
				child = kept[taken]
				taken += 1
			state, detail = self.tracker.advance(base, tid, child is not None)
			if child is not None:
				self.tracker.keep(child.uid, state)
			self.nodes += 1
			cands.append(dict(
				uid=child.uid if child is not None else None, parent=parent.uid, rank=rank_i,
				token=self.tk.tokens[tid] if 0 <= tid < len(self.tk.tokens) else '<unknown>',
				logprob=_r(total - parent.logprob), cum=_r(total), key=_r(key),
				kept=child is not None, eos=(tid == self.tk.eos_id), tick=state['tick'],
				si=_r(state['si'], 6),
				# The align loss the SEARCH ranked on, distinct from `align` below: that is the verdict
				# on a note this token closed (available only at a pitch), this is the forecast the
				# elapse decision was actually taken on. Null where the aligner ABSTAINED; the key is
				# omitted entirely on a run that had no adjudicator, so the two are told apart.
				**(dict(loss=_r(loss, 6)) if self.adjudicated else {}),
				align=None if detail is None else dict(
					src=detail.get('src'), self_cost=_r(detail.get('self_cost')),
					cost=_r(detail.get('cost')), offset=_r(detail.get('offset')),
					skip=detail.get('skip'),
					# How many times this lineage had ALREADY paired this source note. Dumped because a
					# degenerate run is invisible in every other figure -- a re-match reads as
					# self_cost 0.0, skip 0, cost 0.0, i.e. as a PERFECT match, which is exactly how
					# the first pitch-adjudicated run looked while it was emitting one endless chord
					# over 4 distinct source notes.
					reuse=detail.get('reuse'), pitch=detail.get('pitch'),
					onset=detail.get('onset'), softIndex=_r(detail.get('softIndex'), 6),
					src_onset=detail.get('src_onset'), src_pitch=detail.get('src_pitch'))))
		self.window['positions'].append(dict(position=position, candidates=cands,
			live=[dict(uid=b.uid, logprob=_r(b.logprob),
				kind=BRANCH_NAMES[b.state.branch_kind()]) for b in live]))
		# A lineage nothing points at cannot be reached again. Pruning keeps the table proportional to
		# the beam width instead of to every node ever created.
		self.tracker.prune({b.uid for b in kept} | {b.uid for b in done})

	def dump (self, path, meta):
		meta = dict(meta, nodes=self.nodes, truncated=self.truncated,
			windows=len(self.windows), simultaneous_ticks=Config['SIMULTANEOUS_TICKS'],
			reuse_cost=Config['ReuseCost'])
		payload = dict(meta=meta, windows=self.windows,
			source=[dict(onset=e['onset'], pitch=e['pitch'], softIndex=_r(e['softIndex'], 6))
				for e in self.src_events])
		with open(path, 'w', encoding='utf-8') as f:
			json.dump(payload, f, separators=(',', ':'))
		return path


def path_align_loss (tracker, ids):
	'''Accumulated align loss over the PITCH nodes of one finished token sequence.

	The number the search's pitch branch was ranking on, summed along the path it actually took:
	`AlignAdjudicator._pitch_losses` scores a candidate as `detail['cost']`, and this walks the winning
	ids through the SAME `tracker.advance`, so the per-note figures are the adjudicator's own rather
	than a second implementation of them that could drift.

	Computed after the fact rather than accumulated during the search, for two reasons. It works at any
	width and any --rank, including `--beam 1` (which takes the greedy code path and never reaches the
	beam observer), so an align run and its baseline produce comparable numbers. And it cannot perturb
	what it measures: no hook, no extra state on the hypotheses, nothing that runs before a cut.

	Sound because the alignment is a function of the token sequence alone -- `observe` reads a note's
	pitch, tick and softIndex, all of which the ids determine -- so replaying the winner reproduces the
	state the winning lineage held. It also sidesteps the windowing question the live path has to
	handle: this walks the piece once, end to end, so a primer cannot be folded in twice.

	Returns dict(cost, self_cost, notes, final, matched, misses), or None if the walk closed no note.

	`cost` is the requested accumulation and is the quantity the search ranked on, but note what it is:
	`align.observe` keeps `cost` as a DECAYED RUNNING SUM over the recent past, so a note's own
	contribution is re-counted, geometrically attenuated, in every later note. The sum is therefore a
	well-defined and monotone functional of the path -- comparable between two runs over the same
	source -- and NOT a decomposition into per-note charges. `self_cost` is that decomposition (it is
	the only quantity attributable to one note, align.py:734), so both are reported: they answer
	different questions and disagreeing with each other is informative rather than a defect.
	'''
	state = tracker.fresh()
	total = self_total = 0.0
	notes = 0
	for tid in ids:
		state, detail = tracker.advance(state, int(tid))
		if detail is None:
			continue
		notes += 1
		total += detail['cost']
		# None on a miss: `observe` charges an unmatched note through `cost` and attributes no
		# self_cost to it, so the two sums are over different populations by construction. Reported
		# side by side with the miss count, which is what makes that legible.
		if detail.get('self_cost') is not None:
			self_total += detail['self_cost']
	if not notes:
		return None
	align = state['align']
	return dict(cost=total, self_cost=self_total, notes=notes, final=align.cost,
		matched=align.matched, misses=align.misses)


def _r (x, n=5):
	'''Round for the dump, passing None through: a miss has no offset, and 0.0 would read as one.'''
	return None if x is None else round(float(x), n)


def report_beam (report, steps):
	'''What the search actually did, so the branch policy stays checkable rather than asserted.'''
	if not report:
		return
	positions = report.get('steps', 0)
	# branch_points sums over live beams (up to B per position); branch_positions is the per-position
	# count. Reporting only the first and dividing it by positions yields a "rate" above 1, which is
	# how this line first read -- it looked like every position branched four times.
	beam_points = report.get('branch_points', 0)
	branch_positions = report.get('branch_positions', 0)
	kinds = report.get('kinds') or {}
	print(f'[beam] {positions} decode positions over {steps} window(s); '
		f'{branch_positions} of them branched ('
		+ ', '.join(f'{k} {v}' for k, v in sorted(kinds.items())) + ' summed over beams)'
		+ f'; {report.get("expanded", 0)} candidates expanded, '
		f'{report.get("finished", 0)} hypotheses reached <eos>')
	if positions:
		print(f'[beam] branch rate {branch_positions / positions:.3f} of positions, '
			f'{beam_points / positions:.2f} beam-branches and '
			f'{report.get("expanded", 0) / positions:.2f} candidates per position')
		# The margin is a DEFAULT, so a run that pruned nothing and a run built before the margin
		# existed produce the same tree and the same counts -- this line is the only thing that tells
		# them apart in a log.
		if report.get('margin_pruned'):
			print(f"[beam] {report['margin_pruned']} candidates pruned by the logprob margin "
				f'(rated too far below their own row\'s best to be worth a slot)')
	# Only when an adjudicator ran, and it has THREE outcomes per position, not two: it ordered the
	# pool (`adjudicated`), it was asked and its verdicts were recorded while the model still ranked
	# (`scored_only`, i.e. --rank lm --inspect), or it had no evidence (`abstained`). Keying this line
	# on adjudicated/abstained alone reported a score-only run as "the LM-only run" while 2309 of its
	# candidates carried a real loss.
	adjudicated = report.get('adjudicated', 0)
	scored_only = report.get('scored_only', 0)
	abstained = report.get('abstained', 0)
	total = adjudicated + scored_only + abstained
	if total:
		evidenced = adjudicated + scored_only
		if adjudicated:
			overturned = report.get('overturned', 0)
			print(f'[rank] alignment adjudicated {adjudicated} of {total} positions '
				f'({adjudicated / total:.1%}), abstained at {abstained} for want of evidence; '
				f"overturned the model's top candidate at {overturned} "
				f'({overturned / adjudicated:.1%} of adjudicated)')
		elif scored_only:
			# The verdicts are in the dump but changed nothing: this is the model's own tree, annotated.
			print(f'[rank] alignment SCORED {scored_only} of {total} positions '
				f'({scored_only / total:.1%}) and abstained at {abstained}; the model ranked at every '
				f'one of them, so the tree is the LM tree with the aligner\'s verdicts recorded on it')
		else:
			print(f'[rank] alignment abstained at all {abstained} positions: no ratio or residual was '
				f'ever established, so this run is the LM-only run')
		print(f'[rank] {report.get("forced_elapse", 0)} elapse candidates were enumerated that the '
			f"model's top-k did not contain")
		# Reported separately from `abstained` on purpose: these are branches the aligner was never
		# ASKED about, because the lineage had not yet reached its first note_on and an elapse token
		# after it. A run that shows a large number here has spent its opening on the model alone --
		# which is the intent, but it is the kind of intent that should be visible rather than inferred
		# from a flat `adjudicated` count.
		pre_align = report.get('pre_align', 0)
		if pre_align:
			print(f'[rank] {pre_align} branch points before the piece\'s first note_on and the elapse '
				f'token after it were left to the model: no header opinion, and no tick ratio yet')


def main ():
	ap = argparse.ArgumentParser(
		description='Beam-search translation of a midiseq2 file (greedy tool: translateMidiseq2.py).')
	ap.add_argument('--run', default=DEFAULT_RUN, help='training run dir (.state.yaml + checkpoint)')
	ap.add_argument('--checkpoint', default=None,
		help="checkpoint path (default: config['best'], then best.chkpt, then latest.chkpt)")
	ap.add_argument('--input', required=True, help='source .midiseq2.txt')
	ap.add_argument('--output', default=None, help='destination .midiseq2.txt')
	ap.add_argument('--beam', type=int, default=4,
		help='beam width (default 4). 1 falls through to the greedy code path, byte-for-byte')
	ap.add_argument('--branch-k', type=int, default=4,
		help='candidates enumerated per beam at an elapse or pitch position (default 4). Every other '
			'position takes its argmax, so this is the tree width, not the batch width')
	ap.add_argument('--length-alpha', type=float, default=0.7,
		help='length-normalisation exponent when comparing FINISHED hypotheses (default 0.7); '
			'0 = raw logprob, which favours whichever window ended soonest')
	ap.add_argument('--rank', choices=['lm', 'align'], default='align',
		help="candidate ranking (default align): 'lm' = model logprob only (ablation level 2), and an inspection run still RECORDS the alignment's verdicts, it just does not rank on them. 'align' = level 3, "
			'the alignment forecast orders the candidates at an elapse point and the model is the '
			'tie-break')
	ap.add_argument('--elapse-k', type=int, default=8,
		help='with --rank align, how many elapse tokens are enumerated per branch point, taken as '
			'the top-k AMONG elapse tokens rather than from the overall top-k (default 8). The '
			"overall top-k routinely contains no elapse at all, which is why this is separate")
	ap.add_argument('--branch-first', action='store_true',
		help='branch at each window\'s FIRST generated position too. Off by default, i.e. that position '
			'takes the model argmax: the aligner has no pairs there and abstains outright, so the extra '
			'slots go to continuations nothing can adjudicate -- MEASURED at position 0, the argmax was '
			'`ticks_per_beat` at logprob -0.00000 while the other three slots took tokens rated 16.77, '
			'18.05 and 19.11 nats worse, every one with loss None. This flag restores the old behaviour '
			'for ablation')
	ap.add_argument('--align-from-first-token', action='store_true',
		help='let the aligner rank from a window\'s first position instead of waiting for the piece\'s '
			'first note_on AND the first elapse token after it. Off by default, for two reasons. Before '
			'a note_on the file is still its header, and the alignment has no view on whether a header '
			'gets written at all -- MEASURED, with ranking from position 0 the winning lineage descended '
			'from `Eb40` at logprob -19.11, a token that skips the header entirely, and it scored well '
			'only because compose_output substitutes the source header when the body carries none. And '
			'before the first elapse the aligner cannot have a view at all: `_update_ratio` fits a slope '
			'behind `if slope > 0`, so with every target tick still 0 no ratio forms and forecast returns '
			'None -- MEASURED, all 47 elapse branches on the committed dump\'s winning chain abstained. '
			'This flag restores the old behaviour for ablation')
	ap.add_argument('--logprob-margin', type=float, default=2.0,
		help='drop any candidate rated more than this many nats below its OWN ROW\'s best (default '
			'2.0; 0 or less disables it). `--branch-k` is a fixed width, so without this the search '
			'spends the full width at every branch point however certain the model is -- MEASURED at '
			'the committed dump\'s first post-header elapse branch, the four rows put '
			'-0.0000/-0.0000/-0.0002/-0.0003 on elapse-zero while the next candidate was 8.97 nats or '
			'worse, so three of four slots per row went to continuations already ruled out. Per ROW, so '
			'a beam the model likes less overall is not pruned to a narrower width than its neighbours; '
			'a row can never be emptied, since the argmax\'s own margin is 0; and it cannot fire at '
			'width 1, so greedy/beam-1 byte-identity is safe by construction. The adjudicator\'s '
			'forced elapse proposals are EXEMPT -- they exist because the model ranked them badly, so a '
			'margin over them would delete the aligner\'s whole input')
	ap.add_argument('--reuse-cost', type=float, default=None,
		help='charge for a match that RE-USES a source note this lineage already paired, per prior '
			'pairing (align.Config ReuseCost, default 0.0 = off, which is the historical behaviour). '
			'`observe` charges `skip` only for jumping too far AHEAD, so re-using a note was free; '
			'harmless while the pitch is ranked by the language model, an attractor once the pitch is '
			'adjudicated, because re-use is then the cheapest thing on offer -- advancing changes the '
			'offset and self_cost charges the drift, while standing still keeps it at 0. Deliberately '
			'NOT an index-order charge: a chord may be emitted in any order, and 7 of 8 non-advancing '
			'steps on the score-only path were exactly that')
	ap.add_argument('--max-token', type=int, default=2048,
		help='total-T ceiling; pass the training max_tokens (default 2048)')
	ap.add_argument('--src-window', type=int, default=640,
		help='source-half token budget; RE-MEASURE per run (on the [64,512] l16d256 run use 960)')
	ap.add_argument('--no-prime', action='store_true',
		help="don't seed the target half with the previous window's tail")
	ap.add_argument('--advance-tokens', type=int, default=1,
		help='minimum target tokens retired per step, rounded up to the next <eom> (default 1)')
	ap.add_argument('--prime-window', type=int, default=2048,
		help='internal target-view safety ceiling in tokens; not the step stride (default 2048)')
	ap.add_argument('--inspect', nargs='?', type=int, const=0, default=None, metavar='N',
		help='dump the search tree + per-candidate alignment scores as JSON. Bare --inspect records '
			'the whole run; --inspect N records only the FIRST window and caps it at N decode '
			'positions, which is the size a tree view can actually be read at')
	ap.add_argument('--inspect-json', default=None,
		help='where to write it (default: alongside the output, .beamtree.json)')
	ap.add_argument('--max-steps', type=int, default=0, help='stop after N windows (0 = whole file)')
	ap.add_argument('--seed', type=int, default=0)
	ap.add_argument('--threads', type=int, default=0)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--verbose', action='store_true')
	args = ap.parse_args()


	if args.threads:
		torch.set_num_threads(args.threads)
	torch.manual_seed(args.seed)

	# Set BEFORE the tracker or any AlignState exists: Config is read per call inside observe, but
	# setting it after a run has folded notes in would make one file's early notes priced differently
	# from its late ones.
	if args.reuse_cost is not None:
		Config['ReuseCost'] = args.reuse_cost
		print(f'[rank] reuse cost {args.reuse_cost} (a match is charged tanh(prior_pairings * this) '
			f'for re-using a source note; a fresh note is free even out of index order)')

	# Setup follows translateMidiseq2.main step for step, including the validations and the EncDec id
	# assignment. Paraphrasing it is how a beam run ends up on a different pos_style or a differently
	# configured model than the greedy run it is being compared against, which would make a parity
	# result meaningless -- and did: an earlier version of this function mis-read load_model's return
	# and silently handed a Configuration to the translator as its model.
	config = Configuration.createOrLoad(args.run, volatile=True)
	checkpoint = resolve_checkpoint(args.run, config, args.checkpoint)
	config, model = load_model(args.run, checkpoint, args.device)

	data_args = config['data.args'] or {}
	pos_style = data_args.get('pos_style', 'flat')
	model_type = config['model.type']
	if model_type not in ('MidiTranslator', 'MidiTranslatorEncDec'):
		print(f'[error] unsupported model type {model_type!r} for midiseq2 translation')
		return 1
	if pos_style not in ('flat', 'sep', 'absolute'):
		print(f'[error] unsupported data.args.pos_style {pos_style!r}')
		return 1
	trained_max = data_args.get('max_tokens')
	if trained_max and args.max_token != trained_max:
		print(f'[note] --max-token {args.max_token} differs from training max_tokens {trained_max}')
	if data_args.get('source_eom'):
		print('[note] config has source_eom on; the source half will carry <eom> tokens')

	tokenizer, vocab_path = resolve_tokenizer(args.run, config)
	# Bare EncDec inference skips the Loss wrapper, which normally copies these ids from the asset.
	if model_type == 'MidiTranslatorEncDec':
		model.sep_id = tokenizer.sep_id
		model.eos_id = tokenizer.eos_id
		model.pad_id = tokenizer.pad_id
	if vocab_path:
		print(f'[vocab] {vocab_path} ({tokenizer.vocab_size} tokens)')

	with open(args.input, 'r', encoding='utf-8') as f:
		lines = f.read().splitlines()
	print(f'[in]  {os.path.basename(args.input)}: {len(lines)} lines, '
		f'pos_style {pos_style}, src_window {args.src_window}, max_token {args.max_token}')

	# --inspect N means "one window, N positions": the cap applies to GENERATION, not just to what is
	# recorded, so an inspection run costs N positions instead of generating a whole file to keep a
	# fragment of it. Bare --inspect (N == 0) leaves the run alone and records all of it.
	max_steps = args.max_steps
	# ONE lineage table, shared by whichever of the two consumers exist. Built here because both need
	# the same source walk and the same per-hypothesis clocks: giving them a table each is how two
	# tick counters over the same token stream drift apart without either being wrong on its own.
	#
	# Built UNCONDITIONALLY, where it used to be gated on --inspect or --rank align. The final
	# align-loss line is the point: a number only an align run can produce is not comparable to
	# anything, and the baseline it has to be read against is `--rank lm`. What it costs on a run that
	# would not otherwise have one is the source walk here -- no per-candidate AlignState clone happens
	# unless an adjudicator or an inspector asks for one, and neither is created below on an lm run.
	keywords = keyword_tokens(tokenizer)
	src_ids = encode_lines(lines, tokenizer, bool(data_args.get('source_eom')))
	src_events, _abst, _st = note_on_events(src_ids, tokenizer, keywords)
	for e, si in zip(src_events, soft_indices([e['onset'] for e in src_events])):
		e['softIndex'] = si
	tracker = LineageTracker(tokenizer, keywords, src_events)

	# An inspection run ALWAYS scores: a dump exists to answer "what did the aligner think of the tree
	# the model built", and withholding the verdict unless it also ranks would leave every candidate
	# unexplained. --rank align is the separate decision to let those verdicts order the pool, which
	# builds a different tree. So: adjudicator whenever either is wanted, `adjudicate` only for align.
	adjudicate = args.rank == 'align'
	adjudicator = None
	if adjudicate or args.inspect is not None:
		if adjudicate and args.beam <= 1:
			print('[note] --rank align with --beam 1 changes nothing: width 1 has no candidate to '
				'reorder, and the greedy code path is taken')
		adjudicator = AlignAdjudicator(tracker, tokenizer, elapse_k=args.elapse_k)
		print(f'[rank] alignment {"adjudication on (it ORDERS the pool)" if adjudicate else "scoring on (recorded only; the model still ranks)"}, '
			f'elapse-k {args.elapse_k} '
			f'(proposed from {len(adjudicator.proposal_ids)} of the vocabulary\'s '
			f'{len(adjudicator.elapse_values)} elapse tokens; the 15 LOW ones are never proposed, '
			f'and STAGE_MID is not a branch point at all), '
			f'forecast lookahead {Config["ForecastLookahead"]}')

	inspector = None
	if args.inspect is not None:
		if args.beam <= 1:
			print('[note] --inspect with --beam 1 records a tree of width 1; there is nothing to '
				'compare at a position')
		inspector = BeamInspector(tracker, limit=args.inspect, adjudicated=adjudicator is not None)
		if args.inspect:
			max_steps = 1
		print(f'[inspect] recording {"the first window, " + str(args.inspect) + " positions" if args.inspect else "every window"}'
			f'; {len(inspector.src_events)} source note_on to align against')

	Translator = BeamEncDecTranslator if model_type == 'MidiTranslatorEncDec' else BeamTranslator
	translator = Translator(model, tokenizer, pos_style=pos_style, src_window=args.src_window,
		max_token=args.max_token, device=args.device, prime=not args.no_prime,
		temperature=0.0, source_eom=bool(data_args.get('source_eom')),
		advance_tokens=args.advance_tokens, prime_window=args.prime_window,
		beam_size=args.beam, branch_k=args.branch_k, length_alpha=args.length_alpha,
		inspector=inspector, position_cap=args.inspect or 0,
		adjudicator=adjudicator, elapse_k=args.elapse_k if adjudicator is not None else 0,
		adjudicate=adjudicate, greedy_first=not args.branch_first,
		align_from_first_elapse=not args.align_from_first_token,
		logprob_margin=(args.logprob_margin if args.logprob_margin > 0 else None))
	print('[beam] logprob margin ' + (f'{args.logprob_margin:g} nats below each row\'s own best is '
		'pruned' if args.logprob_margin > 0 else 'disabled (the full branch-k everywhere)'))
	print(f'[beam] width {args.beam}, branch-k {args.branch_k} at elapse/pitch, '
		f'{"BRANCHING at" if args.branch_first else "greedy at"} each window\'s first position, '
		f'length-alpha {args.length_alpha}, rank {args.rank}'
		+ ('   (width 1 = the greedy code path)' if args.beam <= 1 else ''))
	# The other opening policy, on the same line of reasoning: it is a default, so a log that does not
	# name it cannot be told apart from one produced before it existed.
	if adjudicator is not None:
		print('[beam] align ranking ' + ('from each window\'s first position (ablation)'
			if args.align_from_first_token
			else 'held off until the piece\'s first note_on AND the elapse token after it: '
				'no header opinion, and no ratio to forecast with before the tick moves'))

	output_ids, stats = translator.translate(lines, verbose=args.verbose, max_steps=max_steps)
	print(f'[out] {stats["steps"]} steps, {stats["output_tokens"]} tokens, '
		f'{stats["consumed_lines"]}/{stats["source_lines"]} source lines consumed, '
		f'{stats["elapsed"]:.1f}s'
		+ (f', {stats["stalls"]} stall(s)' if stats['stalls'] else '')
		+ (f', {stats["eos_forced"]} forced <eos>' if stats['eos_forced'] else '')
		+ ('' if stats['done'] else ', DID NOT reach end_of_track'))
	report_beam(translator.beam_report, stats['steps'])

	body = render_lines(output_ids, tokenizer, translator.keywords)
	out_lines = compose_output(body, source_header(lines))
	out_path = args.output or os.path.join(REPO_ROOT, 'tests', 'output', 'translate_midiseq2',
		os.path.splitext(os.path.basename(args.input))[0] + f'.beam{args.beam}.txt')
	write_output(out_path, out_lines)
	report_output(body, stats)
	print(f'[out] wrote {out_path}')

	# The accumulated align loss of the sequence that was actually emitted, printed after the file so
	# the run's last word is what it produced and what that cost. Reported for EVERY rank, so an align
	# run has a baseline: this walk depends on the ids and the source alone, not on how they were
	# chosen. See `path_align_loss` on why `cost` and `self_cost` are both here -- `cost` is the
	# quantity the pitch branch ranked on and is a decayed running sum, `self_cost` is the per-note
	# decomposition, and they are not two estimates of one number.
	metric = path_align_loss(tracker, output_ids)
	if metric is None:
		print('[align] no note_on in the output, so there is no aligned pitch to accumulate over')
	else:
		print(f'[align] accumulated pitch align loss {metric["cost"]:.4f} over {metric["notes"]} '
			f'note_on ({metric["cost"] / metric["notes"]:.4f} per note), '
			f'self_cost sum {metric["self_cost"]:.4f}, final running cost {metric["final"]:.4f}, '
			f'{metric["matched"]} matched / {metric["misses"]} missed of '
			f'{len(tracker.src_events)} source note_on')

	if inspector is not None:
		json_path = args.inspect_json or os.path.splitext(out_path)[0] + '.beamtree.json'
		meta = dict(run=os.path.basename(args.run.rstrip('/')), checkpoint=os.path.basename(checkpoint),
			# The FULL path, plus token count and mtime: a basename cannot identify an input, and two
			# files named I-YIgmEZ0ss.midiseq2.txt (a 2505-token corpus entry and a 1024-token excerpt)
			# have already been mistaken for each other across two dumps. n_source cannot separate them
			# either -- it caps at src_window + 1 for any input over the window.
			input=os.path.abspath(args.input), input_tokens=sum(len(l.split()) for l in lines),
			input_mtime=int(os.path.getmtime(args.input)), beam=args.beam, branch_k=args.branch_k,
			length_alpha=args.length_alpha, src_window=args.src_window, max_token=args.max_token,
			advance_tokens=args.advance_tokens, pos_style=pos_style, rank=args.rank,
			# The search policies, recorded because they are DEFAULTS: a dump that predates them looks
			# identical in every other key while having been built by a different search, and one such
			# dump has already been read as if it were current.
			greedy_first=not args.branch_first,
			align_from_first_elapse=not args.align_from_first_token,
			logprob_margin=(args.logprob_margin if args.logprob_margin > 0 else None),
			# Whether an aligner RAN is not derivable from `rank`: --rank lm --inspect scores every
			# candidate and merely declines to rank on it. Keying either of the next two on `rank`
			# reported elapse_k None on a run whose pool held 2748 forced elapse candidates, and made
			# the dump-convention check demand that a scored run carry no loss key.
			adjudicator=(None if adjudicator is None else ('ranking' if adjudicate else 'scoring')),
			elapse_k=(args.elapse_k if adjudicator is not None else None),
			positions=sum(len(w['positions']) for w in inspector.windows))
		inspector.dump(json_path, meta)
		size = os.path.getsize(json_path)
		print(f'[inspect] {meta["positions"]} positions, {inspector.nodes} nodes over '
			f'{len(inspector.windows)} window(s) -> {json_path} ({size / 1e6:.2f} MB)')
		print(f'[inspect] view it: open tests/midi/beam_tree_viz.html and drop the JSON on it')
	return 0


if __name__ == '__main__':
	sys.exit(main())
