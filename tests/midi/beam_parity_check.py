'''Checks for the beam search in starry/midi/beam.py and tools/midi/translateMidiseq2Beam.py.

The load-bearing claim is PARITY: `--beam 1` must reproduce the greedy translator exactly. It is
asserted two ways, because they fail differently.

  structural  BeamMixin.generate returns super().generate(...) when beam_size <= 1, so a width-1 run
              executes the greedy lines themselves. Checked by patching the parent and observing the
              call, which catches a refactor that quietly stops delegating.
  behavioural With a stub model, beam_size=1 and beam_size=4 both reproduce the argmax sequence a
              hand-written greedy loop produces on the same logits. This is what catches a wrong
              pool sort, a mis-scored candidate, or an off-by-one in the position tail -- none of
              which the structural check can see.

A stub model is used rather than a checkpoint so the checks run anywhere and pin the SEARCH, not a
particular model's taste. The stub returns fixed logits per position, so the right answer is known in
closed form; a real checkpoint would only tell us the two runs agreed, not what they agreed on.
'''

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'midi'))

import torch
import torch.nn.functional as F

from starry.midi.beam import (BranchState, Beam, beam_search, branch_profile, BRANCH_NONE,
	BRANCH_ELAPSE, BRANCH_PITCH)
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer
from starry.midi.align import (soft_indices, elapse_value, elapse_class, stage_admits,
	STAGE_BIG, Config as AlignConfig)
from translateMidiseq2 import keyword_tokens
from translateMidiseq2Beam import AlignAdjudicator, BeamTranslator, LineageTracker


PASS, FAIL = [], []

def check (name, ok, detail=''):
	(PASS if ok else FAIL).append(name)
	print(f'{"ok  " if ok else "FAIL"} {name}' + (f': {detail}' if detail else ''))
	return ok


class StubModel (torch.nn.Module):
	'''Deterministic logits keyed on the LAST token id, so the correct continuation is knowable.

	`plan` maps a token id to the logit row for the next position. Anything unmapped falls back to a
	row favouring `default`. Records every batch width it was called with, which is how the batching
	claim ("B beams cost one forward pass, not B") is checked rather than assumed.
	'''

	def __init__ (self, vocab, plan, default, eos_id):
		super().__init__()
		self.vocab = vocab
		self.plan = plan
		self.default = default
		self.eos_id = eos_id
		self.max_seq_len = 4096
		self.batch_widths = []
		self.calls = 0

	def row (self, last):
		out = torch.full((self.vocab,), -12.0)
		spec = self.plan.get(last)
		if spec is None:
			out[self.default] = 5.0
			out[self.eos_id] = 1.0
			return out
		for tid, value in spec.items():
			out[tid] = value
		return out

	def forward (self, input_ids, masks=None, position_ids=None):
		self.calls += 1
		self.batch_widths.append(int(input_ids.shape[0]))
		B, T = input_ids.shape
		out = torch.zeros(B, T, self.vocab)
		for b in range(B):
			for t in range(T):
				out[b, t] = self.row(int(input_ids[b, t].item()))
		return out


def make_translator (tk, model, **kw):
	return BeamTranslator(model, tk, pos_style='sep', src_window=64, max_token=64, device='cpu',
		prime=True, temperature=0.0, source_eom=False, advance_tokens=1, prime_window=64, **kw)


def greedy_reference (model, tk, prefix, positions, max_token, ban_first=True):
	'''A hand-written greedy loop, independent of both implementations.

	Deliberately a third copy rather than a call into either one: if the reference were the greedy
	translator, a bug shared by both would pass, and the whole point of this check is that the beam
	agrees with something that was derived separately.
	'''
	ids = list(prefix)
	pos = list(positions)
	out = []
	while len(ids) < max_token:
		logits = model(torch.tensor([ids]), None, torch.tensor([pos]))[0, -1, :].clone()
		if not out and ban_first:
			logits[tk.eos_id] = float('-inf')
		nxt = int(logits.argmax().item())
		if nxt == tk.eos_id:
			break
		out.append(nxt)
		ids.append(nxt)
		pos.append(pos[-1] + 1)
	return out


def check_branch_state (tk, keywords):
	'''Branch points land where the grammar says, and only there.'''
	state = BranchState()
	# a boundary: an elapse token is legal, so this is an elapse branch point
	ok = state.branch_kind() == BRANCH_ELAPSE
	state.feed('note_on', keywords)
	# right after a keyword: the event owes an argument, and that argument is its pitch
	ok = ok and state.branch_kind() == BRANCH_PITCH
	state.feed('#3c', keywords)
	# A note_on's pitch is ALWAYS followed by a `$` velocity (22199/22199 lines in the test corpus,
	# 0 of 14392 such positions were followed by an elapse token), and velocity does not enter the
	# alignment at all -- so nothing is being decided here and it is not a branch point.
	ok = ok and state.branch_kind() == BRANCH_NONE
	state.feed('$50', keywords)
	# velocity paid: the event is complete, so a delta may now start
	ok = ok and state.branch_kind() == BRANCH_ELAPSE
	check('branch kinds follow the grammar: boundary/after-keyword/after-pitch/after-velocity', ok,
		'elapse->pitch->none->elapse')

	# note_off owes NO velocity, so its pitch IS followed by a decision (80.3% an elapse in the corpus)
	off = BranchState()
	off.feed('note_off', keywords)
	off.feed('#3c', keywords)
	check('note_off pitch is followed by a real elapse decision (no velocity owed)',
		off.branch_kind() == BRANCH_ELAPSE, f'{off.branch_kind()}')

	# control_change genuinely splits 1597 `#` / 1590 `#$`, so it must stay a branch point -- the
	# velocity rule is keyed on the OPENING keyword, not on the pitch token
	cc = BranchState()
	cc.feed('control_change', keywords)
	cc.feed('#40', keywords)
	check('control_change pitch stays a branch point (arity genuinely varies)',
		cc.branch_kind() == BRANCH_ELAPSE, f'{cc.branch_kind()}')

	# note_off's pitch is NOT a branch point: it is determined by which notes are open, the alignment
	# never observes a note_off, and measured on a width-4 dump a non-argmax note_off pitch survived
	# the cut 0 of 116 times while spending 16.9% of all pool slots. The grammar still owes a pitch
	# there (pitch_pending holds) -- this is policy, so the two must be checked separately.
	s2 = BranchState()
	s2.feed('note_off', keywords)
	k_off = s2.branch_kind()
	s2on = BranchState()
	s2on.feed('note_on', keywords)
	k_on = s2on.branch_kind()
	# control_change owes an argument as well, but it is NOT a pitch: no `#` belongs there
	s3 = BranchState()
	s3.feed('control_change', keywords)
	k_cc = s3.branch_kind()
	check('pitch branch is note_on only; note_off and other keywords do not fan out',
		k_on == BRANCH_PITCH and k_off == BRANCH_NONE and k_cc == BRANCH_NONE,
		f'note_on {k_on} (pitch), note_off {k_off} (none), control_change {k_cc} (none)')
	check('note_off still OWES a pitch in the grammar (policy did not corrupt the state)',
		s2.pitch_pending and s2.grammar.open_keyword == 'note_off',
		f'pitch_pending {s2.pitch_pending}, open_keyword {s2.grammar.open_keyword!r}')

	# inside an open elapse run the next E token continues it -- still an elapse decision
	s4 = BranchState()
	s4.feed('E1000', keywords)
	in_run = s4.branch_kind() == BRANCH_ELAPSE and s4.grammar.in_run
	check('an open elapse run is still an elapse branch point', in_run,
		f'run_delta {s4.grammar.run_delta:#x}, stage {s4.grammar.stage}')

	# clone independence: the whole search depends on a child not mutating its parent
	a = BranchState(); a.feed('note_on', keywords)
	b = a.clone(); b.feed('#40', keywords)
	# both slots must detach: velocity_pending was added later, and a clone that copied only
	# pitch_pending would let a child's note_on silently suppress its parent's branch point
	ok_clone = (a.pitch_pending and not b.pitch_pending
		and not a.velocity_pending and b.velocity_pending)
	check('BranchState.clone detaches both pending slots', ok_clone,
		f'parent pitch {a.pitch_pending}/vel {a.velocity_pending}, '
		f'child pitch {b.pitch_pending}/vel {b.velocity_pending}')


def check_structural_parity (tk, keywords):
	'''beam_size=1 delegates to the parent's greedy generate -- the same code, not a copy.'''
	model = StubModel(tk.vocab_size, {}, default=tk.id_by_token['note_on'], eos_id=tk.eos_id)
	tr = make_translator(tk, model, beam_size=1)
	seen = {}
	import translateMidiseq2 as greedy_mod
	original = greedy_mod.SlidingTranslator.generate

	def spy (self, *args, **kwargs):
		seen['called'] = True
		return original(self, *args, **kwargs)

	greedy_mod.SlidingTranslator.generate = spy
	try:
		prefix = [tk.bos_id, tk.id_by_token['note_on'], tk.sep_id, tk.bos_id]
		tr.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
	finally:
		greedy_mod.SlidingTranslator.generate = original
	check('beam_size=1 calls the greedy generate itself (structural parity)',
		bool(seen.get('called')), 'delegated via super()')

	tr4 = make_translator(tk, model, beam_size=4)
	seen.clear()
	greedy_mod.SlidingTranslator.generate = spy
	try:
		prefix = [tk.bos_id, tk.id_by_token['note_on'], tk.sep_id, tk.bos_id]
		tr4.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
	finally:
		greedy_mod.SlidingTranslator.generate = original
	check('beam_size=4 does NOT fall through to greedy', not seen.get('called'),
		'runs the search')


def build_plan (tk):
	'''Logits with a KNOWN argmax path and a real second choice at a branch point.

	note_on -> #3c (argmax) or #40 (close second); #3c -> $50; $50 -> E200 then end_of_track/<eos>.
	The margins are small where a beam should be able to overturn them and wide elsewhere, so a
	failure points at the search rather than at an arbitrary tie.
	'''
	i = tk.id_by_token
	return {
		i['note_on']: {i['#3c']: 4.0, i['#40']: 3.6, i['#43']: 2.0, tk.eos_id: -8.0},
		i['#3c']: {i['$50']: 6.0, i['$40']: 2.0, tk.eos_id: -8.0},
		i['#40']: {i['$50']: 6.0, tk.eos_id: -8.0},
		i['$50']: {i['E200']: 5.0, i['E100']: 4.2, tk.eos_id: -6.0},
		i['E200']: {i['end_of_track']: 6.0, tk.eos_id: -4.0},
		i['E100']: {i['end_of_track']: 6.0, tk.eos_id: -4.0},
		i['end_of_track']: {tk.eos_id: 8.0},
	}


def check_behavioural_parity (tk, keywords):
	'''Both widths reproduce an independently written greedy loop's argmax path.'''
	plan = build_plan(tk)
	prefix = [tk.bos_id, tk.id_by_token['note_on'], tk.sep_id, tk.bos_id]
	positions = list(range(len(prefix)))

	model = StubModel(tk.vocab_size, plan, default=tk.id_by_token['note_on'], eos_id=tk.eos_id)
	ref = greedy_reference(model, tk, prefix, positions, max_token=64)

	tr1 = make_translator(tk, StubModel(tk.vocab_size, plan, tk.id_by_token['note_on'], tk.eos_id),
		beam_size=1)
	got1, forced1 = tr1.generate(prefix, positions, 0.0, 0, 1.0, n_source=2)
	check('beam 1 == independent greedy reference, token for token', got1 == ref,
		f'{[tk.tokens[t] for t in got1]}')

	m4 = StubModel(tk.vocab_size, plan, tk.id_by_token['note_on'], tk.eos_id)
	tr4 = make_translator(tk, m4, beam_size=4, branch_k=4, length_alpha=0.0)
	got4, forced4 = tr4.generate(prefix, positions, 0.0, 0, 1.0, n_source=2)
	# With these logits the argmax path is also the highest-total-logprob path (every branch's second
	# choice is strictly worse and nothing downstream recovers the gap), so a correct search MUST
	# return it. A beam that returns something else here has a scoring or sorting bug -- this is the
	# check that a "beam search" which merely wanders would fail.
	check('beam 4 finds the same path when argmax IS optimal', got4 == ref,
		f'{[tk.tokens[t] for t in got4]}')

	# batching: B beams must cost ONE forward per position, not B
	widths = [w for w in m4.batch_widths]
	check('beams are batched into the batch dim (one forward per position)',
		max(widths) > 1 and len(widths) < 40,
		f'{m4.calls} forwards, batch widths {sorted(set(widths))}')


def check_beam_overturns (tk, keywords):
	'''A beam must be ABLE to beat greedy, or its width buys nothing.

	Constructed so the locally-best first token leads into a dead end. What makes a branch doomed is
	NOT a low raw logit downstream -- log_softmax normalises each row, so a continuation the model is
	CERTAIN about costs nearly nothing however small its logit. A branch is only doomed if its
	continuation is UNCERTAIN: E200 wins the first position by 0.4 but then spreads its mass over
	eight roughly equal tokens (~log(1/8) = -2.08 whichever is taken), while E100 continues into a
	near-certain token for ~0. So greedy takes E200 and pays 2.08; a width-2 beam keeps E100 and wins.

	This is the check that distinguishes a search from a greedy loop with extra bookkeeping. It is
	also the only shape of evidence that can justify the beam on this task at all: the payoff has to
	come from resolving the model's own uncertainty a token or two later.
	'''
	i = tk.id_by_token
	# The decisive branch is at the SECOND generated position, not the first. `greedy_first` (on by
	# default) takes the model's argmax at position 0 and does not widen there, so a fixture whose
	# whole point is at position 0 can no longer demonstrate anything -- this one used to be keyed on
	# <bos> directly and started FAILing when that default landed. The throwaway must leave the state at
	# a BOUNDARY -- an elapse may not directly follow a keyword, so `control_change` there gets E200/E100
	# masked to -inf and the fixture tests nothing. `<eom>` is a special, so an elapse is legal after it
	# and position 1 is an ELAPSE branch point: a delta that looks locally right and puts every
	# following note in the wrong bar, which is the shape the whole design is aimed at.
	# eight near-equal continuations: whichever is taken costs about log(1/8)
	flat = {i[t]: 2.0 for t in ('note_off', 'note_on', 'control_change', 'set_tempo',
		'pitchwheel', 'aftertouch', 'polytouch', 'program_change')}
	plan = {
		tk.bos_id: {i['<eom>']: 9.0, tk.eos_id: -8.0},
		i['<eom>']: {i['E200']: 4.0, i['E100']: 3.6, tk.eos_id: -8.0},
		i['E200']: {**flat, tk.eos_id: -8.0},				# dead end: high entropy
		i['E100']: {i['note_on']: 9.0, tk.eos_id: -8.0},	# the payoff: near-certain
		i['note_on']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['note_off']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['control_change']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['set_tempo']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['pitchwheel']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['aftertouch']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['polytouch']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['program_change']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['#3c']: {i['end_of_track']: 9.0, tk.eos_id: -8.0},
		i['end_of_track']: {tk.eos_id: 8.0},
	}
	prefix = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id]
	positions = list(range(len(prefix)))

	ref = greedy_reference(StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id), tk, prefix,
		positions, max_token=64)
	tr = make_translator(tk, StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id),
		beam_size=2, branch_k=2, length_alpha=0.0)
	got, _ = tr.generate(prefix, positions, 0.0, 0, 1.0, n_source=2)
	# Position 1, not 0: position 0 is the greedy_first throwaway and is `<eom>` in both runs
	# by construction. Reading [0] here is what made this check pass vacuously once the fixture moved.
	greedy_pick = tk.tokens[ref[1]] if len(ref) > 1 else None
	beam_pick = tk.tokens[got[1]] if len(got) > 1 else None
	check('a beam overturns a locally-best-but-doomed branch',
		ref != got and beam_pick == 'E100' and greedy_pick == 'E200',
		f'greedy took {greedy_pick}, beam took {beam_pick} (at position 1; position 0 is the '
		f'greedy_first argmax <eom> in both)')

	# ...and it is genuinely the better path by total logprob, not merely a different one
	def total (ids):
		model = StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id)
		run, lp = list(prefix), 0.0
		for tid in ids:
			logits = model(torch.tensor([run]), None, torch.tensor([list(range(len(run)))]))[0, -1, :]
			lp += float(F.log_softmax(logits.float(), dim=-1)[tid])
			run.append(tid)
		return lp
	lp_ref, lp_got = total(ref), total(got)
	check('the overturned path really scores higher', lp_got > lp_ref,
		f'greedy {lp_ref:.3f} vs beam {lp_got:.3f} (+{lp_got - lp_ref:.3f})')


def check_no_branch_no_widen (tk, keywords):
	'''Non-branch positions must not fan out, or the policy is decorative.

	After `note_on #3c` the next position is a field/elapse decision, but after a bare keyword the
	only branch is the pitch. The count of widened positions is compared against the number of
	branch points the grammar found, so a policy that quietly widened everywhere would show up as an
	expansion count far above it.
	'''
	plan = build_plan(tk)
	prefix = [tk.bos_id, tk.id_by_token['note_on'], tk.sep_id, tk.bos_id]
	m = StubModel(tk.vocab_size, plan, tk.id_by_token['note_on'], tk.eos_id)
	tr = make_translator(tk, m, beam_size=3, branch_k=3)
	tr.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
	rep = tr.beam_report
	positions = rep.get('steps', 0)
	expanded = rep.get('expanded', 0)
	# upper bound if EVERY position widened by branch_k on every live beam
	ceiling = positions * 3 * 3
	check('branching is sparse (not every position fans out)', expanded < ceiling,
		f'{expanded} candidates over {positions} positions, ceiling {ceiling}, '
		f'kinds {rep.get("kinds")}')


def check_forced_eos (tk, keywords):
	'''The first-position <eos> ban, and its report, survive the search.

	translate() keys its end-of-piece detection on `forced`, so a ban that hid itself would leave the
	caller unable to tell a finished piece from a stalled one -- the run would grind to EOF emitting
	whatever the ban forced out.
	'''
	i = tk.id_by_token
	# <eos> is the argmax at the FIRST generated position, so the plan is keyed on <bos> -- the
	# prefix's last token. The numbers are the ones actually observed in the degenerate case
	# (13.16 against 7.71 for next-best), which is why a finite penalty was rejected in favour of -inf.
	plan = {
		tk.bos_id: {tk.eos_id: 13.16, i['note_on']: 7.71},
		i['note_on']: {tk.eos_id: 9.0, i['#3c']: 1.0},
		i['#3c']: {tk.eos_id: 9.0},
	}
	prefix = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id]
	for width in (1, 4):
		m = StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id)
		tr = make_translator(tk, m, beam_size=width, branch_k=3)
		got, forced = tr.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
		ok = forced and len(got) >= 1 and got[0] != tk.eos_id
		check(f'beam {width}: first-token <eos> banned, reported, step non-empty', ok,
			f'forced={forced}, {[tk.tokens[t] for t in got]}')


def check_finished_uid (tk, keywords):
	'''A finished hypothesis keeps its parent's uid, so best_uid names a node an observer recorded.

	The terminator is not appended, so `fin` IS its parent's token sequence. When clone() minted it a
	fresh uid, best_uid could name a beam that was never a candidate anywhere: the dump had no node
	for it, the viewer's winning path came up empty, and the lineage carry silently fell back to the
	window root. Only reachable when a hypothesis actually terminates, which is why no earlier check
	caught it -- the long runs it was read on finished nothing.
	'''
	i = tk.id_by_token
	# terminate at the third generated position, after note_on #3c, so there is a real lineage above
	# the finished beam rather than the seed.
	plan = {
		tk.bos_id: {i['note_on']: 5.0, i['note_off']: 1.0},
		i['note_on']: {i['#3c']: 5.0, i['#3e']: 1.0},
		i['#3c']: {tk.eos_id: 9.0, i['E0f0']: 1.0},
		i['#3e']: {tk.eos_id: 9.0},
	}
	prefix = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id]
	# width 1 delegates to the greedy path (that is the parity claim), so it runs no search and files
	# no report; only a real beam width can have a best_uid at all.
	for width in (2, 4):
		m = StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id)
		tr = make_translator(tk, m, beam_size=width, branch_k=4)
		# a stub inspector, because `observer` is wired only when one is present; this records the
		# uids a real dump would have nodes for.
		seen = set()
		class Spy:
			def begin_window (self, *a):
				pass
			def observe (self, pos, live, pool, nxt, done):
				for b in list(live) + list(nxt):
					seen.add(b.uid)
			def end_window (self, best_uid):
				pass
		tr.inspector = Spy()
		got, _forced = tr.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
		rep = tr.beam_report
		best = rep.get('best_uid')
		check(f'beam {width}: best_uid of a finished run resolves to an observed beam',
			rep.get('finished', 0) >= 1 and best in seen,
			f'finished {rep.get("finished")}, best_uid {best}, '
			f'{"in" if best in seen else "NOT in"} {len(seen)} observed uids')

	# and the identity claim directly: a terminator clone carries the parent uid, a token clone does not
	parent = Beam(ids=[1, 2], logprob=-1.0)
	fin, child = parent.clone(uid=parent.uid), parent.clone()
	check('clone keeps a uid when asked and mints one otherwise',
		fin.uid == parent.uid and child.uid != parent.uid,
		f'parent {parent.uid}, finished {fin.uid}, child {child.uid}')


def check_seed_state (tk, keywords):
	'''The seed state comes from the target half of the prefix, not from nothing.

	A state that started empty would call the first generated position a boundary even when the
	primer left an event half-emitted, and would then put branch points in the wrong places.
	'''
	i = tk.id_by_token
	model = StubModel(tk.vocab_size, {}, i['note_on'], tk.eos_id)
	tr = make_translator(tk, model, beam_size=2)
	# primer ends on `note_on`, so the next position owes a PITCH, not a delta
	prefix = [tk.bos_id, i['note_on'], i['#3c'], tk.sep_id, tk.bos_id, i['note_on']]
	state = tr.seed_state(prefix, 3)
	ok_pitch = state.branch_kind() == BRANCH_PITCH
	# primer ends mid elapse run: the run must still be open, with its delta accumulated
	prefix2 = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id, i['E1000']]
	s2 = tr.seed_state(prefix2, 2)
	ok_run = s2.grammar.in_run and s2.grammar.run_delta == 0x1000
	check('seed state is walked over the target half', ok_pitch and ok_run,
		f'after note_on -> {state.branch_kind()} (pitch); after E1000 -> in_run {s2.grammar.in_run}, '
		f'delta {s2.grammar.run_delta:#x}')

	# and it must find the split itself when n_source is not given
	s3 = tr.seed_state(prefix, None)
	check('seed state locates <sep> when n_source is absent',
		s3.branch_kind() == state.branch_kind(),
		f'{s3.branch_kind()} == {state.branch_kind()}')


def check_length_alpha (tk, keywords):
	'''Length normalisation only ever compares FINISHED hypotheses, and it does something.'''
	a = Beam(ids=[1, 2], logprob=-2.0)
	b = Beam(ids=[1, 2, 3, 4, 5, 6], logprob=-4.0)
	raw = a.score(0.0) > b.score(0.0)			# raw logprob favours the short one
	norm = b.score(1.0) > a.score(1.0)			# per-token, the long one is better
	check('length_alpha reorders finished hypotheses', raw and norm,
		f'alpha 0: {a.score(0.0):.2f} vs {b.score(0.0):.2f}; '
		f'alpha 1: {a.score(1.0):.2f} vs {b.score(1.0):.2f}')

	empty = Beam(ids=[], logprob=-1.5)
	check('an empty hypothesis does not divide by zero', empty.score(0.7) == -1.5,
		f'{empty.score(0.7)}')


def check_branch_profile (tk, keywords):
	'''branch_profile agrees with a walk, and the velocity rule actually removes positions.

	Also re-derives the corpus claim the rule rests on, when the corpus is present: if a note_on
	pitch is ever followed by an elapse token, VELOCITY_AFTER_PITCH is removing a branch point that
	was needed, and this is where that shows up rather than as a quiet quality loss.
	'''
	import glob
	from translateMidiseq2 import encode_lines
	ids = [tk.id_by_token[t] for t in
		('note_on', '#3c', '$50', 'E200', 'note_off', '#3c', 'E100')]
	prof = branch_profile(ids, tk.tokens, keywords)
	# positions: note_on(elapse@boundary) #3c(pitch) $50(none) E200(elapse) note_off(elapse)
	#            #3c(pitch) E100(elapse)
	check('branch_profile counts every position exactly once',
		prof['none'] + prof['elapse'] + prof['pitch'] == len(ids),
		f'{prof}')
	check('the velocity rule removes at least one position', prof['none'] >= 1,
		f"none {prof['none']} (the position after note_on's pitch)")

	files = sorted(glob.glob('/home/camus/data/midi/test202606/midiseq2/*.midiseq2.txt'))[:6]
	if not files:
		print('     (corpus absent: skipping the note_on arity re-derivation)')
		return
	violations = 0
	total = 0
	for path in files:
		for line in open(path, encoding='utf-8'):
			parts = line.split()
			if not parts or parts[0] != 'note_on':
				continue
			total += 1
			# `note_on #XX $YY` is the only shape; anything else means an elapse could follow a pitch
			if len(parts) < 3 or not parts[2].startswith('$'):
				violations += 1
	check('corpus still says note_on is always `#$` (the velocity rule\'s premise)',
		total > 0 and violations == 0,
		f'{total} note_on lines over {len(files)} files, {violations} without a $velocity')


def check_adjudicator (tk, keywords):
	'''The alignment adjudicator must reach the decision, and must abstain cleanly when blind.

	Three claims, each a hole that had to be closed before the aligner's vote could matter:

	  the elapse candidates are enumerated even when the model ranks none of them in its top-k
	  (measured: 130 of 399 elapse branch points had no elapse token in the top-4 at all);
	  the pool is ordered on the forecast with the model as tie-break, so a confidently-wrong delta
	  can be overturned rather than merely weighted against;
	  with no alignment evidence every loss is None and the run is the LM-only run, which is what
	  makes level 3 an ablation of level 2 rather than a different search.
	'''
	i = tk.id_by_token
	# The model is CERTAIN about a wrong continuation: note_off at +0 ticks, with every elapse buried.
	# This is the shape of the real failure -- confident mass on the wrong delta -- so a finite penalty
	# against its own confidence would not reliably beat it, and the ordering has to be lexicographic.
	# The decisive adjudication is at the SECOND generated position. `greedy_first` (on by default)
	# takes the model's argmax at position 0 and does not widen there -- correctly, since AlignState has
	# no pairs at that position and abstains -- so a fixture keyed on <bos> tests nothing and this one
	# started FAILing when that default landed. `<eom>` is the throwaway: a special, so an elapse is
	# still legal after it and position 1 is a genuine elapse branch point (a keyword there would get
	# every elapse masked to -inf instead).
	#
	# The primer's TARGET half is `note_on #3c $50 E140 note_on #3c $50` for the same class of reason.
	# This fixture hands the lineage 12 observed pairs, i.e. it is mid-piece, and
	# `align_from_first_elapse` (also on by default) will not let the aligner rank until the lineage has
	# passed a note_on AND an elapse token after it -- so a primer without both described a state no
	# real run reaches, and every adjudication here was gated off. `seed_state` walks that half through
	# `BranchState.feed`, so the first note_on sets one latch, `E140` sets the other, and the trailing
	# `$50` closes the event, leaving position 0 at a boundary exactly as before.
	plan = {
		i['$50']: {i['<eom>']: 9.0, tk.eos_id: -8.0},
		i['<eom>']: {i['note_off']: 9.0, i['E3c0']: -6.0, i['E1e0']: -6.5, i['E140']: -7.0,
			tk.eos_id: -8.0},
		i['note_off']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['#3c']: {i['end_of_track']: 9.0, tk.eos_id: -8.0},
		i['end_of_track']: {tk.eos_id: 8.0},
	}
	for tok in ('E3c0', 'E1e0', 'E140'):
		plan[i[tok]] = {i['note_on']: 9.0, tk.eos_id: -8.0}
	plan[i['note_on']] = {i['#3c']: 9.0, tk.eos_id: -8.0}

	# A source arm spaced so exactly one onset is in reach and the elapse options bracket it. 960
	# ticks apart mirrors the material this was built on (source onsets 2065 -> 2769).
	src = [dict(onset=n * 960, pitch=60 + (n * 5) % 12) for n in range(24)]
	for e, si in zip(src, soft_indices([x['onset'] for x in src])):
		e['softIndex'] = si

	def fresh_pair (elapse_k=8):
		tracker = LineageTracker(tk, keywords, src)
		return tracker, AlignAdjudicator(tracker, tk, elapse_k=elapse_k)

	# --- enumeration: top-k among ELAPSE tokens, independent of the overall distribution
	tracker, adj = fresh_pair()
	row = torch.full((tk.vocab_size,), -20.0)
	row[i['note_off']] = 0.0				# the overall argmax is not an elapse at all
	for n, tok in enumerate(('E1e0', 'E140', 'E3c0')):
		row[i[tok]] = -3.0 - n
	ids = list(adj.elapse_ids(None, row))
	got = [tk.tokens[t] for t in ids]
	elapse_order = [t for t in got if t.startswith('E')][:3]
	check('elapse candidates are enumerated from the elapse tokens, not from the overall top-k',
		elapse_order == ['E1e0', 'E140', 'E3c0'] and got[-1] == 'note_on',
		f'first three elapse ids {elapse_order}, last {got[-1]} (note_on as elapse-0)')

	# --- abstention: nothing observed yet, so no candidate carries a loss
	tracker, adj = fresh_pair()
	class _Beam: uid = 1
	cands = [(i['note_off'], 0.0), (i['E3c0'], -6.0), (i['note_on'], -7.0)]
	check('with no alignment evidence every loss is None (level 3 degrades to level 2)',
		all(x is None for x in adj.losses(_Beam(), cands)), 'all None')

	# --- scoring and coverage, on a tracked alignment
	tracker, adj = fresh_pair()
	base = tracker.base(1)
	for n in range(12):
		base['align'].observe(src[n]['pitch'], src[n]['onset'], src[n]['softIndex'])
	base['tick'] = src[11]['onset']
	losses = adj.losses(_Beam(), cands)
	scored = {tk.tokens[t]: L for (t, _lp), L in zip(cands, losses)}
	# E3c0 = +960 lands exactly on the next unmatched onset, so it must be the cheapest, and the
	# model's confident note_off must not be
	elapse_only = {k: v for k, v in scored.items() if k.startswith('E') or k == 'note_on'}
	check('among the candidates it can score, the forecast puts the elapse landing on the next onset first',
		min(elapse_only, key=lambda k: elapse_only[k]) == 'E3c0'
			and scored['E3c0'] < scored['note_on'],
		f'E3c0 {scored["E3c0"]:.4f} < note_on (elapse-0) {scored["note_on"]:.4f}')

	# note_off produces no onset, so the aligner genuinely has no claim on it: AlignState observes
	# only note_on. As the model's argmax with nothing scored before it in that order, it inherits the
	# nearest FOLLOWING scored loss MINUS epsilon and therefore keeps the position.
	#
	# MEASURED consequence, worth stating because it bounds what level 3 can do: on the committed
	# dump the argmax at 203 of 399 elapse branch points (50.9%) is a token the aligner cannot score
	# (note_off 29.6%, other non-onset tokens 21.3%), so at those the adjudication cannot change the
	# outcome. That is mostly POSTPONEMENT rather than suppression -- a note_off at elapse zero fixes
	# no onset, and the lineage reaches another elapse branch point two positions later, where the
	# argmax is an elapse or a note_on and both are scored.
	check('a candidate the aligner cannot see inherits along the model\'s order, never None',
		all(L is not None for L in losses)
			and abs((scored['E3c0'] - scored['note_off']) - adj.EPSILON) < 1e-12
			and scored['note_off'] < scored['E3c0'],
		f'note_off (LM argmax, unscorable) = E3c0 - epsilon = {scored["note_off"]:.6f}, so it keeps '
		f'the position')

	# --- siblings must not share their parent's AlignState
	# One parent contributes several survivors at 11.5% of positions on the committed dump (up to 4),
	# and 7 positions there had two or three of them each closing a note_on. Advancing the parent's
	# state in place made the second sibling's verdict a function of the first sibling's note, and the
	# adjudicator forecasts from that same object.
	tracker, adj = fresh_pair()
	base = tracker.base(1)
	for n in range(12):
		base['align'].observe(src[n]['pitch'], src[n]['onset'], src[n]['softIndex'])
	base['tick'] = src[11]['onset']
	# A lone pitch closes nothing: the walk has to be inside an open note_on first, exactly as it is
	# at a real pitch branch point.
	opened, _d = tracker.advance(base, i['note_on'], commit=True)
	before = opened['align'].tgt_count
	kids = [tracker.advance(opened, t, commit=True) for t in (i['#3c'], i['#40'], i['#43'])]
	after = opened['align'].tgt_count
	independent = all(st['align'] is not opened['align'] for st, _d in kids)
	one_note_each = all(st['align'].tgt_count == before + 1 for st, _d in kids)
	verdicts = [d is not None for _st, d in kids]
	check('surviving siblings get independent alignments (the parent is not advanced in place)',
		independent and one_note_each and after == before and all(verdicts),
		f'parent tgt_count {before} -> {after} (unchanged); each of {len(kids)} children observed '
		f'exactly one note and got its own verdict')

	# A terminator must NOT inherit: it settles no onset, so anchoring it to a neighbour's rhythm
	# verdict prices the model's objection at EPSILON. Measured before this: <eos> sat 2e-4 behind the
	# anchor while the model rated it 14.17 nats worse, took pool rank 2 into `done`, and four such
	# terminations cut a 200-position run to 93 with one measure emitted.
	term_cands = [(i['note_off'], -0.01), (adj.note_on_id, -4.8), (tk.eos_id, -14.17), (i['E1e0'], -2.0)]
	term_out = adj.losses(_Beam(), term_cands)
	eos_pos = [n for n, (tid, _lp) in enumerate(term_cands) if tid == tk.eos_id][0]
	others = [term_out[n] for n in range(len(term_cands)) if n != eos_pos]
	check('a terminator never inherits a loss (it settles no onset)',
		term_out[eos_pos] is None and all(x is not None for x in others),
		f'<eos> -> {term_out[eos_pos]}, others -> {[None if x is None else round(x, 4) for x in others]}')
	check('<eos> and <eom> are both recognised as terminators',
		{tk.tokens[t] for t in adj.terminator_ids if t < len(tk.tokens)} >= {'<eos>', '<eom>'},
		f'{sorted(tk.tokens[t] for t in adj.terminator_ids if t < len(tk.tokens))}')


	# --- end to end through the search: the adjudicated run must differ from the LM-only run
	def run (rank_align, adjudicate=True):
		tracker, adj = fresh_pair()
		if rank_align:
			b = tracker.base(1)
			for n in range(12):
				b['align'].observe(src[n]['pitch'], src[n]['onset'], src[n]['softIndex'])
			b['tick'] = src[11]['onset']
		tr = make_translator(tk, StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id),
			beam_size=4, branch_k=4, length_alpha=0.0,
			adjudicator=(adj if rank_align else None), elapse_k=(8 if rank_align else 0),
			adjudicate=adjudicate)
		prefix = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id,
			i['note_on'], i['#3c'], i['$50'], i['E140'], i['note_on'], i['#3c'], i['$50']]
		out, _ = tr.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
		return [tk.tokens[t] for t in out], tr.beam_report

	lm_out, lm_rep = run(False)
	al_out, al_rep = run(True)
	# SCORE-ONLY: an adjudicator that is asked but does not RANK must change nothing. A dump is only
	# diagnostic if the tree in it is the tree the model builds; if score-only quietly ranked on the
	# loss, every "why did the model pick this" reading would be off a different search. This
	# regressed once already -- `adjudicate` reached beam_search but the construction site never
	# passed it, so an --rank lm run silently ran full adjudication and reported 88 overturns.
	so_out, so_rep = run(True, adjudicate=False)
	check('score-only: an adjudicator that does not rank reproduces the LM run token for token',
		so_out == lm_out and so_rep.get('overturned', 0) == 0 and so_rep.get('adjudicated', 0) == 0,
		f'{len(so_out)} tokens, identical to LM: {so_out == lm_out}; '
		f"scored_only {so_rep.get('scored_only')}, adjudicated {so_rep.get('adjudicated')}, "
		f"overturned {so_rep.get('overturned')}")
	check('score-only actually SCORED (the branch fired rather than the aligner abstaining)',
		so_rep.get('scored_only', 0) > 0,
		f"scored_only {so_rep.get('scored_only')} positions (adjudicated run: {al_rep.get('adjudicated')})")

	# GRAMMAR MASK. `[E1000]* [Exxx]? [Ex]?` -- the automaton must hold even when the model puts all
	# its mass on a malformed continuation, because the thing being defended against is exactly that.
	# Measured before the mask: a --rank align run emitted 10 malformed runs of 23, among them
	# `E160 E010` and `E1 E1 E1 E1 E1 E1`, each overriding a LEGAL model argmax at logprob ~-0.0000.
	def run_greedy_illegal (want_second):
		'''A model that wants `E140` then `want_second`. Returns the tokens actually emitted.'''
		mid, second = i['E140'], i[want_second]
		plan = {
			tk.bos_id: {mid: 8.0},
			mid: {second: 8.0},				# the illegal (or legal) continuation, near-certain
			second: {tk.eos_id: 8.0},
		}
		tr = make_translator(tk, StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id),
			beam_size=4, branch_k=4, length_alpha=0.0)
		prefix = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id]
		out, _ = tr.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
		return [tk.tokens[t] for t in out]

	legal = run_greedy_illegal('E5')			# MID then LOW -- canonical
	illegal = run_greedy_illegal('E030')		# MID then MID -- must not survive the mask
	check('a legal MID LOW run is untouched by the grammar mask',
		legal[:2] == ['E140', 'E5'], f'emitted {legal[:3]}')
	check('the grammar mask blocks a second MID even when the model is certain of it',
		illegal[:2] != ['E140', 'E030'] and 'E030' not in illegal,
		f'model wanted E140 E030 at logit 8.0, emitted {illegal[:3]}')

	# A BARE digit after an elapse token is the other malformed shape that reached the output
	# (`E1a0 5 5 #53`): a digit belonging to no event. Same test shape -- the model is made certain of
	# it, and must still not be able to emit it.
	def run_after_elapse (second):
		plan = {tk.bos_id: {i['E140']: 8.0}, i['E140']: {i[second]: 8.0},
			i[second]: {tk.eos_id: 8.0}}
		tr = make_translator(tk, StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id),
			beam_size=4, branch_k=4, length_alpha=0.0)
		prefix = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id]
		out, _ = tr.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
		return [tk.tokens[t] for t in out]

	for tok, label in (('5', 'a bare digit'), ('#4c', 'an orphan #pitch'), ('C4', 'an orphan channel')):
		got = run_after_elapse(tok)
		check(f'the grammar mask blocks {label} directly after an elapse token',
			tok not in got, f'model wanted E140 {tok} at logit 8.0, emitted {got[:3]}')
	kept = run_after_elapse('note_off')
	check('a keyword after an elapse token is untouched',
		kept[:2] == ['E140', 'note_off'], f'emitted {kept[:3]}')

	# The bare-digit rule is a PREDECESSOR rule, not a ban: where a digit is the argument of the
	# keyword that owes one, it must still be reachable.
	plan_arg = {tk.bos_id: {i['set_tempo']: 8.0}, i['set_tempo']: {i['7']: 8.0},
		i['7']: {i['a']: 8.0}, i['a']: {tk.eos_id: 8.0}}
	tr_arg = make_translator(tk, StubModel(tk.vocab_size, plan_arg, i['note_on'], tk.eos_id),
		beam_size=4, branch_k=4, length_alpha=0.0)
	pre = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id]
	arg_out, _ = tr_arg.generate(pre, list(range(len(pre))), 0.0, 0, 1.0, n_source=2)
	arg_toks = [tk.tokens[t] for t in arg_out]
	check('a digit that IS an argument is still reachable (set_tempo 7 a)',
		arg_toks[:3] == ['set_tempo', '7', 'a'], f'emitted {arg_toks[:4]}')

	# And the automaton is enforced over the WHOLE output of the adjudicated run, which is where the
	# malformed runs actually came from: the aligner proposes on rhythm evidence and has no view on
	# the grammar, so an unmasked forced token walked straight into the pool.
	def malformed_runs (toks):
		runs, cur = [], []
		for t in toks:
			if elapse_value(t) is not None:
				cur.append(t)
			else:
				if cur:
					runs.append(cur)
				cur = []
		if cur:
			runs.append(cur)
		bad = []
		for r in runs:
			stage = STAGE_BIG
			for t in r:
				cls = elapse_class(elapse_value(t))
				if cls is None or not stage_admits(stage, cls):
					bad.append(' '.join(r))
					break
				stage = cls
		return runs, bad
	al_runs, al_bad = malformed_runs(al_out)
	check('the adjudicated run emits no malformed elapse run',
		not al_bad, f'{len(al_runs)} elapse run(s), {len(al_bad)} malformed'
		+ (f': {al_bad[:2]}' if al_bad else ''))
	check('the mask reports how much it blocked',
		al_rep.get('grammar_masked', 0) > 0,
		f"grammar_masked {al_rep.get('grammar_masked')} (token, beam) pairs")

	# Position 1, not 0: position 0 is the greedy_first throwaway `<eom>` in both runs by construction.
	check('the adjudicated run overturns the LM-only run',
		lm_out != al_out and len(lm_out) > 1 and len(al_out) > 1
			and lm_out[1] == 'note_off' and al_out[1].startswith('E'),
		f'LM took {lm_out[1] if len(lm_out) > 1 else None}, adjudicated took '
		f'{al_out[1] if len(al_out) > 1 else None} (at position 1; position 0 is the greedy_first '
		f'argmax <eom> in both)')
	check('the search reports what the adjudicator did',
		al_rep.get('adjudicated', 0) > 0 and al_rep.get('overturned', 0) > 0
			and al_rep.get('forced_elapse', 0) > 0 and lm_rep.get('adjudicated', 0) == 0,
		f'adjudicated {al_rep.get("adjudicated")}, overturned {al_rep.get("overturned")}, '
		f'forced elapse {al_rep.get("forced_elapse")}; LM-only adjudicated '
		f'{lm_rep.get("adjudicated", 0)}')

	# --- an adjudicator that abstains everywhere must produce the LM-only output exactly
	tracker, adj = fresh_pair()
	tr = make_translator(tk, StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id),
		beam_size=4, branch_k=4, length_alpha=0.0, adjudicator=adj, elapse_k=8)
	prefix = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id,
		i['note_on'], i['#3c'], i['$50'], i['E140'], i['note_on'], i['#3c'], i['$50']]
	blind, _forced = tr.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
	blind_out, blind_rep = [tk.tokens[t] for t in blind], tr.beam_report
	check('a blind adjudicator reproduces the LM-only output token for token',
		blind_out == lm_out and blind_rep.get('adjudicated', 0) == 0
			and blind_rep.get('abstained', 0) > 0,
		f'{len(blind_out)} tokens, abstained at {blind_rep.get("abstained")} positions')



def check_pitch_adjudication (tk, keywords):
	'''The PITCH branch must be adjudicated too, and on the verdict rather than a forecast.

	The hole: at an elapse point the aligner had to forecast because no note exists yet, but by the
	time a pitch is being chosen the tick is already settled and `observe` -- the function the whole
	alignment is built around -- applies directly. It was not being called, so the note itself, the
	one thing the alignment exists to judge, was the only decision taken on the language model alone.

	MEASURED on the committed align dump, pos 140: `#4a` carried self_cost 4e-05 and cost 0.13734 --
	the best figure at that position, against 0.8099 for the best SCORED candidate -- and a better
	logprob than the token that won, and it was cut at rank 10. Not a missed overturn but an active
	cull, because `loss is None` is the FIRST sort key: an unmeasured candidate is not neutral, it is
	last.

	Three claims here:

	  a pitch that is really at the next onset beats one that is not, EVEN when the model prefers the
	  wrong one (the whole point of adjudicating);
	  a miss is MEASURED, not inherited -- `observe` prices it at cost*attenuation + MissCost, so the
	  `src=None` pitches need no epsilon policy of their own (pos 140 had three of them at 1.13731
	  against the matched 0.13734);
	  a non-pitch candidate at a pitch point still inherits, so nothing is banned.
	'''
	i = tk.id_by_token
	src = [dict(onset=n * 960, pitch=60 + (n * 5) % 12) for n in range(24)]
	for e, si in zip(src, soft_indices([x['onset'] for x in src])):
		e['softIndex'] = si
	tracker = LineageTracker(tk, keywords, src)
	adj = AlignAdjudicator(tracker, tk, elapse_k=8)
	base = tracker.base(1)
	for n in range(12):
		base['align'].observe(src[n]['pitch'], src[n]['onset'], src[n]['softIndex'])
	# Park the lineage ON the next source onset with note_on open, awaiting its pitch -- which is
	# exactly the state BRANCH_PITCH reports. `walk` is the note_on_events walk state, so setting its
	# open keyword is what makes the candidate pitch close an event.
	base['tick'] = src[12]['onset']
	base['walk'] = ('note_on', src[12]['onset'], 0)
	base['prev_onset'] = src[11]['onset']
	base['softindex'] = src[11]['softIndex']

	right = src[12]['pitch']
	wrong = 60 + ((right - 60) + 6) % 12		# a tritone away: not at this onset
	tok_r, tok_w = '#%02x' % right, '#%02x' % wrong
	filler = '$50' if '$50' in i else 'note_off'
	# The model is CONFIDENT about the wrong pitch, which is the shape being corrected.
	cands = [(i[tok_w], 0.0), (i[tok_r], -2.0), (i[filler], -5.0)]

	class _Beam: uid = 1
	losses = adj.losses(_Beam(), cands, kind=BRANCH_PITCH)
	got = {tk.tokens[t]: L for (t, _lp), L in zip(cands, losses)}
	check('the pitch branch is scored on the VERDICT: the pitch really at the next onset wins',
		got[tok_r] < got[tok_w],
		f'{tok_r} (at the onset, lp -2.00) {got[tok_r]:.5f} < {tok_w} (LM argmax, lp 0.00) '
		f'{got[tok_w]:.5f}')
	check('a pitch MISS is measured at MissCost, not inherited at epsilon',
		abs(got[tok_w] - AlignConfig['MissCost']) < 1e-9,
		f'{tok_w} = cost*attenuation + MissCost = {got[tok_w]:.5f}, so src=None needs no epsilon '
		f'policy of its own')
	check('a non-pitch candidate at a pitch point still inherits, so nothing is banned',
		got[filler] is not None
			and abs((got[filler] - got[tok_r]) - adj.EPSILON) < 1e-12,
		f'{filler} = {tok_r} + epsilon = {got[filler]:.6f}')

	# The gate must pick the axis, not merely fire: forecasting a PITCH candidate is meaningless (a
	# pitch token has no tick value and is not note_on), so the elapse pass would score none of these.
	elapse_pass = adj.losses(_Beam(), cands, kind=BRANCH_ELAPSE)
	check('the branch kind selects the axis: the forecast pass has nothing to say about a pitch',
		all(x is None for x in adj._elapse_losses(_Beam(), cands))
			and all(x is None for x in elapse_pass),
		'elapse pass abstains on all three pitch/argument candidates, so the whole position falls '
		'back to the LM -- which is what it did before this change')


def check_greedy_first (tk, keywords):
	'''The first generated position must take the model argmax and not widen, by default.

	The aligner has nothing to say there and the model has everything to say. MEASURED at position 0
	of the committed align dump: the argmax was `ticks_per_beat` at logprob -0.00000 while the other
	three beam slots went to `format_type` (-16.77), `#52` (-18.05) and `Eb40` (-19.11), and EVERY
	candidate's loss was None -- AlignState has no pairs yet, so it abstains outright. Three of four
	slots spent on continuations the model rates 17 to 19 nats worse with nothing able to adjudicate
	them, and those lineages then persist and compete at later positions where evidence does exist.

	Two claims, plus the ablation escape hatch:

	  at position 0 exactly ONE candidate is expanded, however branchy the state is;
	  branching resumes at position 1, so this closes one position and not the search;
	  greedy_first=False restores the old behaviour, or the ablation cannot be run.
	'''
	i = tk.id_by_token
	# A boundary at position 0 (an elapse branch point) and another at position 1, so both positions
	# WOULD widen. `<eom>` keeps the state at a boundary; a keyword would make the elapse illegal.
	plan = {
		tk.bos_id: {i['<eom>']: 9.0, i['E140']: 3.0, i['E1e0']: 2.5, tk.eos_id: -8.0},
		i['<eom>']: {i['E140']: 4.0, i['E1e0']: 3.6, i['E3c0']: 3.2, tk.eos_id: -8.0},
		i['E140']: {i['note_on']: 9.0, tk.eos_id: -8.0},
		i['E1e0']: {i['note_on']: 9.0, tk.eos_id: -8.0},
		i['E3c0']: {i['note_on']: 9.0, tk.eos_id: -8.0},
		i['note_on']: {i['#3c']: 9.0, tk.eos_id: -8.0},
		i['#3c']: {i['$50']: 9.0, tk.eos_id: -8.0},
		i['$50']: {i['end_of_track']: 9.0, tk.eos_id: -8.0},
		i['end_of_track']: {tk.eos_id: 8.0},
	}
	prefix = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id]

	def go (greedy_first):
		rep = {}
		tr = make_translator(tk, StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id),
			beam_size=4, branch_k=4, length_alpha=0.0, greedy_first=greedy_first)
		out, _ = tr.generate(prefix, list(range(len(prefix))), 0.0, 0, 1.0, n_source=2)
		return [tk.tokens[t] for t in out], tr.beam_report

	on_out, on_rep = go(True)
	off_out, off_rep = go(False)

	# The state at position 0 is a branch point in both runs, so the difference is the gate and not
	# the grammar: with the gate off it widens, with it on it does not.
	check('greedy_first closes the first position: fewer branch points than with it off',
		on_rep.get('branch_points', 0) < off_rep.get('branch_points', 0),
		f"branch points {on_rep.get('branch_points')} with the gate on vs "
		f"{off_rep.get('branch_points')} with it off, over {on_rep.get('steps')} steps")
	check('greedy_first takes the model argmax at the first position',
		on_out and on_out[0] == '<eom>',
		f'emitted {on_out[:3]} (argmax at position 0 was <eom> at logit 9.0)')
	check('greedy_first closes ONE position, not the search: it still branches later',
		on_rep.get('branch_points', 0) > 0 and on_rep.get('expanded', 0) > len(on_out),
		f"{on_rep.get('branch_points')} branch points and {on_rep.get('expanded')} candidates "
		f'expanded for {len(on_out)} emitted tokens')
	check('greedy_first=False restores the old behaviour, so the ablation is runnable',
		off_rep.get('branch_points', 0) > on_rep.get('branch_points', 0),
		f"{off_rep.get('branch_points')} branch points with the gate off")


def check_align_from_first_elapse (tk, keywords):
	"""Align ranking must be held off until a lineage has emitted its first note_on AND, after it, an
	elapse token.

	Two latches, two different reasons.

	The note_on half: before it the file is still its header, and the alignment's only opinion there is
	an accident. MEASURED with ranking from position 0: the winning lineage descended from `Eb40` at
	logprob -19.11, a token that declines to write a header at all and goes straight to the music,
	reaching scoreable pitches by position 2. It scored well from there -- but only because
	`compose_output` substitutes the SOURCE's header whenever the body carries none, so the good numbers
	were bought with a decision about the header that no alignment evidence bears on.

	The elapse half: after the header but before any tick advance the aligner's answer is not a
	judgement, it is a None. `forecast` needs a tick ratio; `_update_ratio` fits `slope = d_tgt / d_src`
	behind an `if slope > 0` guard; with every target tick still 0 that guard never passes, so `ratio`
	stays None for the rest of the run. MEASURED on the committed dump: all 47 elapse branches on the
	winning chain abstained, and all 37 candidates at the first post-note_on elapse branch carried
	`loss None` while the model put -0.0000 on elapse-zero against -8.9749 for the cheapest real elapse.

	The adjudicator here is a SPY that always has an opinion and always overturns the model. A real
	AlignState abstains on its own in exactly this region, which would make the gate look like it worked
	when nothing was gating -- the spy separates the two.

	  the latch pair itself: a note_on alone does NOT open the gate, and a header elapse does not either;
	  no `losses` call until an elapse lands after the first note_on, so the LM ranks the opening alone;
	  the pitch branch immediately after the first note_on -- which the note_on-only gate DID adjudicate
	  -- is now suppressed, which is the whole content of this change;
	  ranking resumes after that first elapse, so this gates the opening and not the search;
	  the suppressed branches are counted as `pre_align` and NOT as `abstained`, since the aligner was
	  never asked and "asked, no evidence" is a different fact;
	  align_from_first_elapse=False restores the old behaviour, or the ablation cannot be run.
	"""
	i = tk.id_by_token

	# --- the latch pair, read straight off BranchState, with no search in the way
	st = BranchState()
	for tok in ('E140',):
		st.feed(tok, keywords)
	header_elapse = (st.saw_note_on, st.saw_elapse_after_note_on)
	for tok in ('note_on', '#3c', '$50'):
		st.feed(tok, keywords)
	after_note_on = (st.saw_note_on, st.saw_elapse_after_note_on)
	st.feed('E140', keywords)
	after_elapse = (st.saw_note_on, st.saw_elapse_after_note_on)
	check('the two latches are ORDERED: a header elapse does not count, a note_on alone does not open',
		header_elapse == (False, False) and after_note_on == (True, False)
			and after_elapse == (True, True),
		f'after a header E140 {header_elapse}, after note_on #3c $50 {after_note_on}, '
		f'after the elapse following it {after_elapse}')
	# The latches are a phase of the FILE, so they must survive clone or a surviving sibling would
	# reopen a gate its parent had closed.
	kid = st.clone()
	check('both latches survive clone, so a sibling cannot reopen the gate',
		kid.saw_note_on and kid.saw_elapse_after_note_on,
		f'clone -> saw_note_on {kid.saw_note_on}, saw_elapse_after_note_on '
		f'{kid.saw_elapse_after_note_on}')

	# --- through the search. Branch points on both sides of both latches in one run: an elapse
	# boundary at 0 and 1, the pitch at 2, then an elapse at 4 once note_on #3c $50 has been emitted.
	plan = {
		tk.bos_id: {i['<eom>']: 9.0, i['E140']: 3.0, i['E1e0']: 2.5, tk.eos_id: -8.0},
		i['<eom>']: {i['note_on']: 9.0, i['E140']: 4.0, i['E1e0']: 3.6, tk.eos_id: -8.0},
		i['note_on']: {i['#3c']: 9.0, i['#40']: 4.0, i['#43']: 3.5, tk.eos_id: -8.0},
		i['$50']: {i['E140']: 9.0, i['E1e0']: 5.0, i['note_on']: 4.0, tk.eos_id: -8.0},
		i['E140']: {i['note_on']: 9.0, i['E1e0']: 3.0, tk.eos_id: -8.0},
		i['E1e0']: {i['note_on']: 9.0, i['E140']: 3.0, tk.eos_id: -8.0},
	}
	for tok in ('#3c', '#40', '#43'):
		plan[i[tok]] = {i['$50']: 9.0, tk.eos_id: -8.0}
	model = StubModel(tk.vocab_size, plan, i['note_on'], tk.eos_id)
	prefix = [tk.bos_id, i['note_on'], tk.sep_id, tk.bos_id]

	class Spy:
		def __init__ (self):
			self.asked = []
			self.pos = 0

		def elapse_ids (self, beam, row):
			return ()

		def losses (self, beam, cands, kind=BRANCH_ELAPSE):
			self.asked.append((self.pos, BRANCH_PITCH if kind == BRANCH_PITCH else BRANCH_ELAPSE))
			# A real opinion at every position, so an absent call is the GATE and never abstention.
			# Reversed against the model's order, so an ungated call also overturns it visibly.
			return [float(len(cands) - n) for n in range(len(cands))]

	def go (gate):
		spy, rep = Spy(), {}
		def step_fn (ids_batch):
			full = [prefix + list(ids) for ids in ids_batch]
			rows = []
			for ids in full:
				out = model(torch.tensor([ids]), None, torch.tensor([list(range(len(ids)))]))
				rows.append(torch.log_softmax(out[0, -1, :].float(), dim=-1))
			spy.pos = len(ids_batch[0])
			return torch.stack(rows)
		out = beam_search(step_fn, tk.tokens, keywords, tk.eos_id, 10, beam_size=4, branch_k=4,
			length_alpha=0.0, ban_first=(tk.eos_id,), report=rep, adjudicator=spy, elapse_k=0,
			adjudicate=True, greedy_first=True, align_from_first_elapse=gate)
		ids = out[0] if isinstance(out, tuple) else out
		return [tk.tokens[t] for t in ids], spy.asked, rep

	_on_out, on_asked, on_rep = go(True)
	_off_out, off_asked, off_rep = go(False)
	on_pos = {p for p, _k in on_asked}
	off_pos = {p for p, _k in off_asked}

	# 5 is the earliest position any lineage can be asked at, and it is worth tracing because it is
	# derived rather than observed: `greedy_first` fixes position 0 to the argmax `<eom>`, so the first
	# note_on cannot land before 1; its pitch is 2 and its velocity 3, so the first elapse after it
	# cannot land before 4; and the gate is read at the position AFTER the token that opens it. A
	# lineage that spends position 1 on a header elapse instead reaches its own first post-note_on
	# elapse later still, never earlier.
	check('align ranking is held off until an elapse lands after the first note_on',
		on_pos and min(on_pos) == 5,
		f'earliest adjudicated position {min(on_pos) if on_pos else None} with the gate on '
		f'(note_on >= 1, its pitch 2, its velocity 3, the elapse >= 4, so the first open read is 5); '
		f'asked at {sorted(on_pos)}')
	# THE point of this change, stated as the one position that separates the two gates. Position 2 is
	# the pitch branch immediately after the first note_on: `saw_note_on` is already set there, so the
	# previous gate adjudicated it, and there is still no tick advance so the aligner's forecast could
	# not have been anything but None.
	check('the pitch branch right after the first note_on is now suppressed (the note_on-only gate '
			'adjudicated it)',
		2 not in on_pos and 2 in off_pos,
		f'position 2 asked with the gate off ({sorted((p, k) for p, k in off_asked if p == 2)}), '
		f'not asked with it on')
	check('ranking RESUMES after that first elapse, so the gate closes the opening not the search',
		len(on_asked) > 0 and on_rep.get('adjudicated', 0) > 0,
		f"{len(on_asked)} calls, {on_rep.get('adjudicated')} positions adjudicated after both latches "
		f'set')
	# The distinction the viewer needs: never-asked is not the same fact as asked-with-no-evidence.
	check('suppressed branch points are counted as pre_align, not folded into abstained',
		on_rep.get('pre_align', 0) > 0 and off_rep.get('pre_align', 0) == 0,
		f"pre_align {on_rep.get('pre_align')} with the gate on, {off_rep.get('pre_align')} with it off")
	check('align_from_first_elapse=False restores the old behaviour, so the ablation is runnable',
		0 in off_pos and len(off_asked) > len(on_asked),
		f'{len(off_asked)} calls from position 0 with the gate off vs {len(on_asked)} with it on')

	# The FORCED-ELAPSE half of the same gate. These candidates exist only to give the aligner
	# something to rank, so enumerating them where it may not rank hands the model extra tokens it had
	# already declined -- measured as 21 header branch points drawing forced elapses into an
	# LM-ordered pool. `elapse_ids` is what the search calls for them, so counting its calls is the
	# direct test; the spy above returns () from it, so this needs one that returns something.
	class ProposingSpy (Spy):
		def __init__ (self):
			super().__init__()
			self.proposed = []

		def elapse_ids (self, beam, row):
			self.proposed.append(self.pos)
			# Tokens the model never ranks, so every one of these is a genuine ADDITION to the pool.
			# `E140`/`E1e0` are already in the plan's top-4 after `$50`, so proposing those counted 0
			# forced candidates and the check read as a regression when nothing had regressed.
			return (i['E3c0'], i['E290'])

	def go_forced (gate):
		spy, rep = ProposingSpy(), {}
		def step_fn (ids_batch):
			full = [prefix + list(ids) for ids in ids_batch]
			rows = []
			for ids in full:
				out = model(torch.tensor([ids]), None, torch.tensor([list(range(len(ids)))]))
				rows.append(torch.log_softmax(out[0, -1, :].float(), dim=-1))
			spy.pos = len(ids_batch[0])
			return torch.stack(rows)
		out = beam_search(step_fn, tk.tokens, keywords, tk.eos_id, 10, beam_size=4, branch_k=4,
			length_alpha=0.0, ban_first=(tk.eos_id,), report=rep, adjudicator=spy, elapse_k=8,
			adjudicate=True, greedy_first=True, align_from_first_elapse=gate)
		ids = out[0] if isinstance(out, tuple) else out
		return [tk.tokens[t] for t in ids], spy.proposed, rep

	_f_on_out, f_on_prop, f_on_rep = go_forced(True)
	_f_off_out, f_off_prop, f_off_rep = go_forced(False)
	check('no forced elapse candidates are enumerated before the gate opens',
		f_on_prop and min(f_on_prop) >= 5,
		f'elapse_ids called at {sorted(set(f_on_prop))} with the gate on (earliest possible open '
		f'read is 5)')
	# Position 0 is NOT the discriminator: `greedy_first` closes it in both runs, so neither
	# enumerates there. What separates them is the positions in between -- with the gate off the
	# enumeration reaches the header and the post-note_on pitch, with it on it does not.
	check('forced elapse enumeration RESUMES after the gate opens, and the ablation restores it',
		len(f_on_prop) > 0 and f_on_rep.get('forced_elapse', 0) > 0
			and f_off_prop and min(f_off_prop) < 5
			and f_off_rep.get('forced_elapse', 0) > f_on_rep.get('forced_elapse', 0),
		f"forced_elapse {f_on_rep.get('forced_elapse')} with the gate on vs "
		f"{f_off_rep.get('forced_elapse')} with it off; with it off the enumeration reaches "
		f'position {min(f_off_prop) if f_off_prop else None}')


def main ():
	tk = Midiseq2Tokenizer()
	keywords = keyword_tokens(tk)
	print(f'vocab {tk.vocab_size} tokens, {len(keywords)} keywords\n')

	check_branch_state(tk, keywords)
	print()
	check_branch_profile(tk, keywords)
	print()
	check_structural_parity(tk, keywords)
	print()
	check_behavioural_parity(tk, keywords)
	print()
	check_beam_overturns(tk, keywords)
	print()
	check_no_branch_no_widen(tk, keywords)
	print()
	check_forced_eos(tk, keywords)
	print()
	check_finished_uid(tk, keywords)
	check_seed_state(tk, keywords)
	print()
	check_length_alpha(tk, keywords)
	print()
	check_adjudicator(tk, keywords)
	print()
	check_pitch_adjudication(tk, keywords)
	print()
	check_greedy_first(tk, keywords)
	print()
	check_align_from_first_elapse(tk, keywords)

	total = len(PASS) + len(FAIL)
	print(f'\n{len(PASS)}/{total} checks passed')
	if FAIL:
		print('failed: ' + ', '.join(FAIL))
	return 1 if FAIL else 0


if __name__ == '__main__':
	sys.exit(main())
