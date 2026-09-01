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
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from starry.midi.beam import BranchState, beam_search, BRANCH_NAMES

# The greedy translator is the base, not a template: these are the same objects it uses.
from translateMidiseq2 import (DEFAULT_RUN, SlidingTranslator, SlidingEncDecTranslator,
	resolve_checkpoint, resolve_tokenizer, load_model, render_lines, compose_output, write_output,
	report_output, source_header)
from starry.utils.config import Configuration


class BeamMixin:
	'''Adds beam search to a SlidingTranslator by overriding `generate` and nothing else.

	`translate` calls `self.generate(...)`, so a subclass is the whole integration: no sliding-window
	code is duplicated, re-derived, or forked. beam_size=1 falls through to the parent's greedy
	implementation by an explicit early return, so a width-1 run does not merely resemble greedy --
	it IS greedy, running the same lines.
	'''

	def __init__ (self, *args, beam_size=1, branch_k=4, length_alpha=0.7, **kwargs):
		super().__init__(*args, **kwargs)
		self.beam_size = max(1, int(beam_size))
		self.branch_k = max(1, int(branch_k))
		self.length_alpha = length_alpha
		self.beam_report = {}		# accumulated across steps; read after translate() returns

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
		ids, forced, _rep = beam_search(step, self.tk.tokens, self.keywords, self.tk.eos_id,
			max_new=room, beam_size=self.beam_size, branch_k=self.branch_k,
			length_alpha=self.length_alpha,
			seed_state=self.seed_state(list(prefix_ids), n_source),
			# Same first-position terminator ban as the greedy path, for the same measured reason: an
			# <eos> logit of 13.16 against 7.71 for next-best is not something a finite penalty can be
			# tuned against. `forced` must keep flowing back, because translate() keys its
			# end-of-piece detection on it.
			ban_first=(self.tk.eos_id,), rank=self.rank_fn(), report=self.beam_report)
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

	Translator = BeamEncDecTranslator if model_type == 'MidiTranslatorEncDec' else BeamTranslator
	translator = Translator(model, tokenizer, pos_style=pos_style, src_window=args.src_window,
		max_token=args.max_token, device=args.device, prime=not args.no_prime,
		temperature=0.0, source_eom=bool(data_args.get('source_eom')),
		advance_tokens=args.advance_tokens, prime_window=args.prime_window,
		beam_size=args.beam, branch_k=args.branch_k, length_alpha=args.length_alpha)
	print(f'[beam] width {args.beam}, branch-k {args.branch_k} at elapse/pitch, '
		f'length-alpha {args.length_alpha}, rank {args.rank}'
		+ ('   (width 1 = the greedy code path)' if args.beam <= 1 else ''))

	output_ids, stats = translator.translate(lines, verbose=args.verbose, max_steps=args.max_steps)
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
	return 0


if __name__ == '__main__':
	sys.exit(main())
