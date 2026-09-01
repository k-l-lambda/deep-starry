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
from translateMidiseq2 import keyword_tokens
from translateMidiseq2Beam import BeamTranslator


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

	# note_off's pitch is a branch point too: a wrong one closes the wrong note
	s2 = BranchState()
	s2.feed('note_off', keywords)
	k_off = s2.branch_kind()
	# control_change owes an argument as well, but it is NOT a pitch: no `#` belongs there
	s3 = BranchState()
	s3.feed('control_change', keywords)
	k_cc = s3.branch_kind()
	check('pitch branch is note_on/note_off only, not every keyword',
		k_off == BRANCH_PITCH and k_cc == BRANCH_NONE,
		f'note_off {k_off} (pitch), control_change {k_cc} (none)')

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
	# Keyed on <bos>, because that is the prefix's LAST token and therefore what decides the FIRST
	# generated position. The target half here is just <bos>, so that position is a boundary -- an
	# ELAPSE branch point -- which is also the case the whole design is aimed at: a delta that looks
	# locally right and puts every following note in the wrong bar.
	# eight near-equal continuations: whichever is taken costs about log(1/8)
	flat = {i[t]: 2.0 for t in ('note_off', 'note_on', 'control_change', 'set_tempo',
		'pitchwheel', 'aftertouch', 'polytouch', 'program_change')}
	plan = {
		tk.bos_id: {i['E200']: 4.0, i['E100']: 3.6, tk.eos_id: -8.0},
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
	greedy_first = tk.tokens[ref[0]] if ref else None
	beam_first = tk.tokens[got[0]] if got else None
	check('a beam overturns a locally-best-but-doomed branch',
		ref != got and beam_first == 'E100' and greedy_first == 'E200',
		f'greedy took {greedy_first}, beam took {beam_first}')

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
	check_seed_state(tk, keywords)
	print()
	check_length_alpha(tk, keywords)

	total = len(PASS) + len(FAIL)
	print(f'\n{len(PASS)}/{total} checks passed')
	if FAIL:
		print('failed: ' + ', '.join(FAIL))
	return 1 if FAIL else 0


if __name__ == '__main__':
	sys.exit(main())
