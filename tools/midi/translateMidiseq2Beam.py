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

Ablation ladder this is step 2 of (1 = greedy, already the other script):
  2. beam, LM-only ranking            <- HERE. --beam 4 --rank lm
  3. beam + alignment ranking         --rank align   (reorders candidates; changes no token's legality)
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

from starry.midi.beam import BranchState, beam_search, BRANCH_NAMES
from starry.midi.align import AlignState, Config, soft_delta, soft_indices

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
		position_cap=0, **kwargs):
		super().__init__(*args, **kwargs)
		self.beam_size = max(1, int(beam_size))
		self.branch_k = max(1, int(branch_k))
		self.length_alpha = length_alpha
		self.beam_report = {}		# accumulated across steps; read after translate() returns
		self.inspector = inspector
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
			seed_state=self.seed_state(list(prefix_ids), n_source),
			# Same first-position terminator ban as the greedy path, for the same measured reason: an
			# <eos> logit of 13.16 against 7.71 for next-best is not something a finite penalty can be
			# tuned against. `forced` must keep flowing back, because translate() keys its
			# end-of-piece detection on it.
			ban_first=(self.tk.eos_id,), rank=self.rank_fn(), report=self.beam_report)
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

	def __init__ (self, tokenizer, source_lines, source_eom, keywords=None, seed_offset=0.0, limit=0):
		self.tk = tokenizer
		# Derived here rather than taken from the translator: the source walk happens in this
		# constructor, before a translator exists, and a walk with no keywords silently finds no
		# note_ons at all -- an empty source arm against which every generated note is a miss.
		self.keywords = keywords if keywords is not None else keyword_tokens(tokenizer)
		self.limit = limit				# 0 = uncapped; else stop recording after N positions per window
		# The source arm, walked once: its note_ons with softIndex, which is what AlignState matches
		# against. Doing it here rather than per window keeps one source coordinate for the whole run.
		src_ids = encode_lines(source_lines, tokenizer, source_eom)
		self.src_events, _abst, _st = note_on_events(src_ids, tokenizer, self.keywords)
		for e, si in zip(self.src_events, soft_indices([e['onset'] for e in self.src_events])):
			e['softIndex'] = si
		self.seed_offset = seed_offset
		self.windows = []
		self.window = None
		self.lineage = {}				# uid -> lineage dict, pruned to the live set each position
		self.root = self._fresh()
		self.nodes = 0
		self.truncated = False

	def _fresh (self):
		return dict(align=AlignState(self.src_events, seed_offset=self.seed_offset),
			walk=None, tick=0, prev_onset=None, softindex=0.0)

	def _advance (self, base, tid, commit):
		'''Walk ONE token on top of `base` -> (new lineage dict, alignment verdict or None).

		The verdict is None unless this token CLOSED a note_on: an elapse moves the clock but has
		nothing to match yet, and a velocity or channel is not a musical event at all. So a node
		carries an alignment score exactly when it created something the alignment can judge.

		`commit` False scores on a clone and discards it, which is how a cut candidate is scored
		without polluting the lineage that survived.
		'''
		align = base['align'] if commit else base['align'].clone()
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
			softindex=softindex), detail

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
		self.root = self.lineage.get(best_uid, self.root)
		self.window = None

	def observe (self, position, live, pool, kept, done):
		if self.window is None:
			return
		if self.limit and len(self.window['positions']) >= self.limit:
			self.truncated = True
			return
		for beam in live:
			self.lineage.setdefault(beam.uid, self.root)
		# Which pool entries became survivors, by the search's OWN rule: walk the sorted pool, and
		# the first len(kept) non-terminator entries are the ones it kept. Derived rather than
		# matched on token identity -- two beams can propose the same token at the same position, and
		# a match on (token, prefix) would then attach the recording to the wrong lineage.
		taken = 0
		cands = []
		for rank_i, (key, total, row, tid) in enumerate(pool):
			parent = live[row]
			base = self.lineage[parent.uid]
			child = None
			if tid == self.tk.eos_id:
				pass				# went to `done`; it never becomes a live child
			elif taken < len(kept):
				child = kept[taken]
				taken += 1
			state, detail = self._advance(base, tid, child is not None)
			if child is not None:
				self.lineage[child.uid] = state
			self.nodes += 1
			cands.append(dict(
				uid=child.uid if child is not None else None, parent=parent.uid, rank=rank_i,
				token=self.tk.tokens[tid] if 0 <= tid < len(self.tk.tokens) else '<unknown>',
				logprob=_r(total - parent.logprob), cum=_r(total), key=_r(key),
				kept=child is not None, eos=(tid == self.tk.eos_id), tick=state['tick'],
				align=None if detail is None else dict(
					src=detail.get('src'), self_cost=_r(detail.get('self_cost')),
					cost=_r(detail.get('cost')), offset=_r(detail.get('offset')),
					skip=detail.get('skip'), pitch=detail.get('pitch'),
					onset=detail.get('onset'), softIndex=_r(detail.get('softIndex')),
					src_onset=detail.get('src_onset'), src_pitch=detail.get('src_pitch'))))
		self.window['positions'].append(dict(position=position, candidates=cands,
			live=[dict(uid=b.uid, logprob=_r(b.logprob),
				kind=BRANCH_NAMES[b.state.branch_kind()]) for b in live]))
		# A lineage nothing points at cannot be reached again. Pruning keeps the table proportional to
		# the beam width instead of to every node ever created.
		alive = {b.uid for b in kept} | {b.uid for b in done}
		self.lineage = {u: v for u, v in self.lineage.items() if u in alive}

	def dump (self, path, meta):
		meta = dict(meta, nodes=self.nodes, truncated=self.truncated,
			windows=len(self.windows), simultaneous_ticks=Config['SIMULTANEOUS_TICKS'])
		payload = dict(meta=meta, windows=self.windows,
			source=[dict(onset=e['onset'], pitch=e['pitch'], softIndex=_r(e['softIndex'], 6))
				for e in self.src_events])
		with open(path, 'w', encoding='utf-8') as f:
			json.dump(payload, f, separators=(',', ':'))
		return path


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
	ap.add_argument('--rank', choices=['lm', 'align'], default='lm',
		help="candidate ranking: 'lm' = model logprob only (ablation level 2). 'align' is level 3 "
			'and is not wired up yet')
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

	if args.rank == 'align':
		ap.error('--rank align is ablation level 3 and is not implemented yet; use --rank lm')

	if args.threads:
		torch.set_num_threads(args.threads)
	torch.manual_seed(args.seed)

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
	inspector = None
	if args.inspect is not None:
		if args.beam <= 1:
			print('[note] --inspect with --beam 1 records a tree of width 1; there is nothing to '
				'compare at a position')
		inspector = BeamInspector(tokenizer, lines, bool(data_args.get('source_eom')),
			limit=args.inspect)
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
		inspector=inspector, position_cap=args.inspect or 0)
	print(f'[beam] width {args.beam}, branch-k {args.branch_k} at elapse/pitch, '
		f'length-alpha {args.length_alpha}, rank {args.rank}'
		+ ('   (width 1 = the greedy code path)' if args.beam <= 1 else ''))

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

	if inspector is not None:
		json_path = args.inspect_json or os.path.splitext(out_path)[0] + '.beamtree.json'
		meta = dict(run=os.path.basename(args.run.rstrip('/')), checkpoint=os.path.basename(checkpoint),
			input=os.path.basename(args.input), beam=args.beam, branch_k=args.branch_k,
			length_alpha=args.length_alpha, src_window=args.src_window, max_token=args.max_token,
			advance_tokens=args.advance_tokens, pos_style=pos_style, rank=args.rank,
			positions=sum(len(w['positions']) for w in inspector.windows))
		inspector.dump(json_path, meta)
		size = os.path.getsize(json_path)
		print(f'[inspect] {meta["positions"]} positions, {inspector.nodes} nodes over '
			f'{len(inspector.windows)} window(s) -> {json_path} ({size / 1e6:.2f} MB)')
		print(f'[inspect] view it: open tests/midi/beam_tree_viz.html and drop the JSON on it')
	return 0


if __name__ == '__main__':
	sys.exit(main())
