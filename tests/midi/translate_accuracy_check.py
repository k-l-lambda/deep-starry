#!/usr/bin/env python3
'''Score MidiTranslator's irregular->score output against GROUND TRUTH, and sweep --src-window.

Every earlier quality number for this task was self-consistency (does the output parse, does it
carry barlines) or input-overlap (are the pitches the ones we fed in). Neither can say whether the
right note lands at the right time, because neither has the answer to compare against.
test202608/nota1m-100 does: 100 files present in BOTH arms under the same name, so
midi-seq2-irregular/X is the input and midi-seq2-score/X is what the model should have produced.

Reported metrics, deliberately SEPARATE (see --help for why one blended score would mislead):

  pitch F1    multiset of note_on pitches, time ignored. Answers "are these the right notes".
  onset F1    multiset of (tick, pitch). Answers "at the right time" — the actual job of
              irregular->score, and the one that can be near-zero while pitch F1 looks healthy.

Onset F1 is computed twice, because two different defects both depress it and the fix differs:

  absolute    ticks on the file axis. One dropped measure early on shifts everything after it, so
              this is the end-to-end number but it conflates local rhythm with global drift.
  bar-relative  tick measured from the start of its own measure, compared measure by measure.
              Survives drift, so it isolates "wrong rhythm inside the bar".

Alignment is by MEASURE index, not by token position: the model reorders simultaneous onsets
within a chord, so position-wise comparison is the wrong instrument (an earlier session recorded
ordered prefix agreement as 0 and that was an artifact, not a finding). Scoring covers measures
1..min(ref, out) minus the output's last measure, which a --max-steps cut leaves half-generated.

Usage
    # one setting, 8 files
    python3 tests/midi/translate_accuracy_check.py --checkpoint /tmp/pin/e366.chkpt \\
        --corpus /home/camus/data/midi/test202608/nota1m-100 --files 8 --src-window 640 \\
        --device cuda --max-steps 12

    # the sweep the conclusion comes from
    python3 tests/midi/translate_accuracy_check.py --checkpoint /tmp/pin/e366.chkpt \\
        --corpus /home/camus/data/midi/test202608/nota1m-100 --files 8 \\
        --sweep 480,640,960,1260 --device cuda --max-steps 12 --out /tmp/acc_sweep.json

ALWAYS pass --checkpoint explicitly. The hourly weight sync overwrites best.chkpt in the local
mirror and once swapped the model halfway through a sweep, so a comparison table straddled two
epochs. Pin a copy and point at the copy.
'''

import argparse
import json
import os
import sys
import time
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'midi'))

import translateMidiseq2 as T			# noqa: E402


# --- parsing ------------------------------------------------------------------------------

def is_elapse (tok):
	return T.is_elapse(tok)


def parse_notes (lines):
	'''midiseq2 lines -> (notes, measure_starts, tick_annotations).

	notes          list of (abs_tick, pitch, channel, measure_index)
	measure_starts measure_index -> abs_tick of that bar's FIRST note_on (the bar-relative anchor)
	tick_deltas    (my_delta, claimed_delta) between consecutive @tick lines inside one bar, used
	               to VALIDATE the elapse accumulator

	Time comes from accumulating each event's leading `E…` run, the same convention
	seq2CondPachifier.parse_events uses. @tick is redundant in the corpus (the elapse run already
	says it) and absent from model output entirely, so it is never the source of truth here.

	Why the bar anchor is the first note_on and NOT the `@measure` directive: the directive's tick
	is not reliably the downbeat. In this corpus `@measure 41` is followed by an `E1e0` run and the
	bar truly starts 480 ticks later (38400, an exact multiple of the 960-tick 2/4 bar, where the
	directive sat at 37920, which is not) — yet other bars carry the identical shape
	`@measure / elapse / @tick 480`, meaning that bar started BEFORE its elapse. The token shape is
	ambiguous and only the @tick value resolves it, which model output does not have. Anchoring on
	each stream's own first note_on is symmetric, needs no directive semantics, and measures the
	intended thing: given where this bar begins, are the rest of its notes spaced correctly.

	The accumulator is validated instead by within-bar @tick DELTAS, which depend only on elapse
	arithmetic and not on where a bar is deemed to start.

	Measure index counts @measure lines. The corpus opens with `@measure 1`; model output rendered
	by render_lines starts its numbering at 2 because <eom> marks the OPENING of bar N for N>=2 and
	bar 1 emits no token. Both are handled by reading the printed number when present.
	'''
	notes = []
	measure_starts = {}
	deltas = []
	abst = 0
	measure = 1
	prev_tick = None			# (my_abst, claimed) of the previous @tick in THIS bar

	for line in lines:
		toks = line.split()
		if not toks:
			continue
		head = toks[0]
		if head == '@measure':
			measure = int(toks[1]) if len(toks) > 1 and toks[1].lstrip('-').isdigit() else measure + 1
			prev_tick = None		# deltas are only meaningful within one bar
			continue
		# an elapse-only line advances time and opens nothing
		if is_elapse(head):
			for tok in toks:
				if is_elapse(tok):
					abst += int(tok[1:], 16)
			continue
		if head == '@tick':
			if len(toks) > 1 and toks[1].lstrip('-').isdigit():
				claimed = int(toks[1])
				if prev_tick is not None:
					deltas.append((abst - prev_tick[0], claimed - prev_tick[1]))
				prev_tick = (abst, claimed)
			continue
		if head == 'note_on':
			pitch = None
			channel = 0
			for tok in toks[1:]:
				if tok.startswith('#') and pitch is None:
					pitch = int(tok[1:], 16)
				elif tok.startswith('C'):
					try:
						channel = int(tok[1:], 16)
					except ValueError:
						pass
			if pitch is not None:
				# the bar's anchor is its first note_on, in whichever stream we are reading
				measure_starts.setdefault(measure, abst)
				notes.append((abst, pitch, channel, measure))
	return notes, measure_starts, deltas


def pairing_defects (lines):
	'''note_on/note_off hygiene, keyed on (channel, pitch) — the defect an earlier session isolated.

	Returns (note_ons, re_on, orphan_off, stuck). A re-on is a note_on for a (channel, pitch) that
	is already sounding; an orphan is a note_off with nothing sounding; stuck are still held at the
	end. Totals alone hide this: note_on and note_off counts can balance exactly while every pair is
	mismatched, and the output still parses. Real score-arm references score ~0 here, so anything
	large is the model's own defect rather than corpus noise.
	'''
	sounding = set()
	note_ons = re_on = orphan = 0
	for line in lines:
		toks = line.split()
		if not toks or toks[0] not in ('note_on', 'note_off'):
			continue
		pitch = None
		channel = 0
		for tok in toks[1:]:
			if tok.startswith('#') and pitch is None:
				pitch = int(tok[1:], 16)
			elif tok.startswith('C'):
				try:
					channel = int(tok[1:], 16)
				except ValueError:
					pass
		if pitch is None:
			continue
		key = (channel, pitch)
		if toks[0] == 'note_on':
			note_ons += 1
			if key in sounding:
				re_on += 1
			sounding.add(key)
		else:
			if key in sounding:
				sounding.discard(key)
			else:
				orphan += 1
	return note_ons, re_on, orphan, len(sounding)


def f1 (pred_counter, ref_counter):
	'''Multiset F1. Returns (precision, recall, f1, tp, n_pred, n_ref).

	Multiset rather than set: a repeated (tick, pitch) is a real duplicate-note defect and set
	semantics would silently forgive it.
	'''
	tp = sum((pred_counter & ref_counter).values())
	n_pred = sum(pred_counter.values())
	n_ref = sum(ref_counter.values())
	p = tp / n_pred if n_pred else 0.0
	r = tp / n_ref if n_ref else 0.0
	return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, n_pred, n_ref


def quantize (tick, grid):
	return int(round(tick / grid)) * grid if grid > 1 else tick


def score_pair (out_lines, ref_lines, grid, score_bars=0, skip_bars=0, shift_span=0,
	verbose=False):
	'''Compare one output against its ground truth. Returns a dict of metrics, or None if the
	output carries too few measures to score.'''
	out_notes, out_starts, _ = parse_notes(out_lines)
	ref_notes, ref_starts, ref_deltas = parse_notes(ref_lines)

	# Parser self-check on the REFERENCE, which annotates @tick: the time our elapse accumulator
	# advances between two @tick lines in a bar must equal the advance the file itself claims. If
	# this fails the timing model is wrong and every number below is meaningless, so it is an
	# assertion rather than a warning. Deltas rather than absolute positions because `@measure` does
	# not reliably mark the downbeat (see parse_notes) — this tests the arithmetic we depend on and
	# nothing we do not.
	bad = [(mine, claimed) for mine, claimed in ref_deltas if mine != claimed]
	if bad:
		raise AssertionError(f'elapse accumulation disagrees with @tick deltas at '
			f'{len(bad)}/{len(ref_deltas)} points, first {bad[0]}')

	if not out_notes:
		return None
	out_measures = sorted({m for _, _, _, m in out_notes})
	ref_measures = sorted({m for _, _, _, m in ref_notes})
	if len(out_measures) < 2:
		return None
	# Drop the output's last measure: a --max-steps cut or an <eos> mid-bar leaves it partial, and a
	# half-generated bar scores as a precision failure that is an artifact of where we stopped.
	last = out_measures[-1]
	hi = min(last - 1, ref_measures[-1])
	lo = max(min(out_measures), min(ref_measures))
	if hi < lo:
		return None
	# Cap the span to a FIXED bar count so a sweep compares identical music. Without this, a bigger
	# src_window consumes more source per step and therefore covers more bars at the same
	# --max-steps, so the settings would be scored on different content and the comparison would be
	# meaningless. Files that cannot reach the cap are dropped by the caller, not scored short.
	# Skip the opening bars before scoring. Step 0 generates a whole window's worth of target in ONE
	# shot (~6 bars); only later steps exercise the sliding seam, which is where src_window is
	# supposed to matter. Scoring from bar 1 would therefore mostly grade single-shot generation and
	# report almost no window effect regardless of the truth.
	lo += skip_bars
	if score_bars:
		if hi - lo + 1 < score_bars:
			return None
		hi = lo + score_bars - 1
	if hi < lo:
		return None
	span = set(range(lo, hi + 1))

	out_in = [n for n in out_notes if n[3] in span]
	ref_in = [n for n in ref_notes if n[3] in span]
	if not ref_in:
		return None

	pitch_out = Counter(p for _, p, _, _ in out_in)
	pitch_ref = Counter(p for _, p, _, _ in ref_in)
	abs_out = Counter((quantize(t, grid), p) for t, p, _, _ in out_in)
	abs_ref = Counter((quantize(t, grid), p) for t, p, _, _ in ref_in)

	# How far apart the two streams already are AT THE WINDOW'S FIRST BAR. Absolute ticks accumulate
	# from each stream's own start, so any duration error in the SKIPPED bars (1..skip_bars) arrives
	# here as a constant offset and would be charged to every scored bar. Comparing this against the
	# best constant shift below separates the two cases: `anchor_delta ~= shift_ticks` means the
	# offset was INHERITED from bars we deliberately excluded, while a shift that is not predicted by
	# anchor_delta is displacement generated inside the scored window — a live defect.
	anchor_delta = out_starts.get(lo, 0) - ref_starts.get(lo, 0) if lo in out_starts else None
	# Absolute tick rebased on that anchor: timing WITHIN the scored window, free of inherited drift.
	# Reported alongside raw onset_f1 rather than replacing it, since rebasing guarantees the window's
	# first note matches and so flatters the score slightly.
	span_out = (Counter((quantize(t - out_starts[lo], grid), p) for t, p, _, _ in out_in)
		if lo in out_starts else Counter())
	span_ref = Counter((quantize(t - ref_starts.get(lo, 0), grid), p) for t, p, _, _ in ref_in)
	# bar-relative: offset from the start of the note's own measure, so a whole-file drift cancels
	rel_out = Counter((m, quantize(t - out_starts.get(m, 0), grid), p) for t, p, _, m in out_in)
	rel_ref = Counter((m, quantize(t - ref_starts.get(m, 0), grid), p) for t, p, _, m in ref_in)

	# Absolute tick but keyed BY BAR. Sits between the other two and is what makes them readable:
	# bar_f1 vs onset_f1 isolates disagreement about where the barlines go (same instant, different
	# bar number), while rel_f1 vs bar_f1 isolates within-bar spacing given the bar already agrees.
	# Without this, a low rel_f1 cannot be told apart from bad rhythm.
	bar_out = Counter((m, quantize(t, grid), p) for t, p, _, m in out_in)
	bar_ref = Counter((m, quantize(t, grid), p) for t, p, _, m in ref_in)

	on_o, reon_o, orph_o, stuck_o = pairing_defects(out_lines)
	on_r, reon_r, orph_r, stuck_r = pairing_defects(ref_lines)

	pit = f1(pitch_out, pitch_ref)
	ons = f1(abs_out, abs_ref)
	bar = f1(bar_out, bar_ref)
	rel = f1(rel_out, rel_ref)
	spn = f1(span_out, span_ref)

	# Is a low onset F1 "wrong rhythm" or "right music at the wrong absolute tick"? Those need
	# different fixes and the strict metric cannot tell them apart, so search a single constant
	# shift. If one d recovers a high score, the notes and their spacing were right and only the
	# stream's absolute position was off — which also means the strict number is partly grading how
	# well the scored bar SPAN happened to line up, not the model's timing.
	best_shift, best_shift_f1 = 0, ons[2]
	if shift_span:
		notes_out = [(t, p) for t, p, _, _ in out_in]
		for d in range(-shift_span, shift_span + 1, grid):
			cand = Counter((quantize(t + d, grid), p) for t, p in notes_out)
			score = f1(cand, abs_ref)[2]
			if score > best_shift_f1:
				best_shift, best_shift_f1 = d, score
	return dict(
		measures_scored=len(span), measures_out=len(out_measures), measures_ref=len(ref_measures),
		notes_out=len(out_in), notes_ref=len(ref_in),
		pitch_p=pit[0], pitch_r=pit[1], pitch_f1=pit[2],
		onset_p=ons[0], onset_r=ons[1], onset_f1=ons[2],
		shift_f1=best_shift_f1, shift_ticks=best_shift,
		span_f1=spn[2], anchor_delta=anchor_delta,
		bar_p=bar[0], bar_r=bar[1], bar_f1=bar[2],
		rel_p=rel[0], rel_r=rel[1], rel_f1=rel[2],
		reon_rate=(reon_o / on_o if on_o else 0.0), stuck_out=stuck_o, orphan_out=orph_o,
		reon_rate_ref=(reon_r / on_r if on_r else 0.0), stuck_ref=stuck_r,
		channels_out=len({c for _, _, c, _ in out_in}),
		channels_ref=len({c for _, _, c, _ in ref_in}),
	)


# --- driver -------------------------------------------------------------------------------

def mean (values):
	return sum(values) / len(values) if values else 0.0


def setting_label (src_window, prime_window, no_prime=False):
	'''`--no-prime` disables the target prefix entirely, so the prime_window value is meaningless
	there. Labelling such a run `960:320` records a budget it never used and invites a later reader to
	compare it against a real prime-320 run, so say `noprime` instead.
	'''
	return f'{src_window}:{"noprime" if no_prime else prime_window}'


def parse_settings (spec, default_prime):
	'''`--sweep 960,1260:520` -> [(960, default_prime), (1260, 520)].

	Bare entries keep --prime-window so an existing src-only sweep command means exactly what it
	meant before; `src:prime` pins both.
	'''
	settings = []
	for entry in spec.split(','):
		entry = entry.strip()
		if not entry:
			continue
		if ':' in entry:
			src, prime = entry.split(':', 1)
			settings.append((int(src), int(prime)))
		else:
			settings.append((int(entry), default_prime))
	return settings


def run_setting (model, tokenizer, config, pairs, src_window, prime_window, args):
	'''Translate every pair at one (src_window, prime_window) and aggregate.

	`prime_window` is the internal target-view safety ceiling. The public stride is
	`args.advance_tokens`; keeping these axes separate prevents a token-retention
	budget from being mistaken for a movement setting.
	'''
	data_args = config['data.args'] or {}
	translator = T.SlidingTranslator(model, tokenizer,
		pos_style=data_args.get('pos_style', 'flat'),
		src_window=src_window, max_token=args.max_token, device=args.device,
		prime=not args.no_prime, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
		source_eom=bool(data_args.get('source_eom')), advance_tokens=args.advance_tokens,
		prime_window=prime_window)

	rows = []
	for name, src_path, ref_path in pairs:
		with open(src_path, 'r', encoding='utf-8') as f:
			src_lines = f.read().splitlines()
		with open(ref_path, 'r', encoding='utf-8') as f:
			ref_lines = f.read().splitlines()
		t0 = time.time()
		ids, stats = translator.translate(src_lines, verbose=False, max_steps=args.max_steps)
		body = T.render_lines(ids, tokenizer, translator.keywords)
		took = time.time() - t0
		row = score_pair(body, ref_lines, args.grid, score_bars=args.score_bars,
			skip_bars=args.skip_bars, shift_span=args.shift_span)
		if row is None:
			print(f'  {name[:28]:28s}  unscorable (fewer than {args.score_bars or 2} scorable bars)')
			continue
		# `stalls` = steps that generated nothing mid-piece. This is the direct observable for the
		# over-long-prime failure trim_prime documents (the model reads a long target half as already
		# finished and emits <eos> at once), so it is the field that tells a prime_window regression
		# apart from ordinary quality loss.
		row.update(name=name, seconds=took, steps=stats['steps'],
			stalls=stats['stalls'], output_tokens=stats['output_tokens'])
		rows.append(row)
		anchor = 'n/a' if row['anchor_delta'] is None else f'{row["anchor_delta"]:+d}'
		print(f'  {name[:28]:28s}  bars {row["measures_scored"]:3d}  '
			f'pitchF1 {row["pitch_f1"]:.3f}  onsetF1 {row["onset_f1"]:.3f}  '
			f'barF1 {row["bar_f1"]:.3f}  relF1 {row["rel_f1"]:.3f}  '
			f'reon {row["reon_rate"]*100:4.1f}%  '
			f'spanF1 {row["span_f1"]:.3f}  anchor {anchor}  '
			f'shifted {row["shift_f1"]:.3f}@{row["shift_ticks"]:+d}  '
			f'stall {row["stalls"]:d}  {took:5.1f}s')

	if not rows:
		return rows, None
	agg = dict(src_window=src_window,
		prime_window=None if args.no_prime else prime_window, no_prime=bool(args.no_prime),
		label=setting_label(src_window, prime_window, args.no_prime), files=len(rows),
		pitch_f1=mean([r['pitch_f1'] for r in rows]),
		pitch_p=mean([r['pitch_p'] for r in rows]), pitch_r=mean([r['pitch_r'] for r in rows]),
		onset_f1=mean([r['onset_f1'] for r in rows]),
		onset_p=mean([r['onset_p'] for r in rows]), onset_r=mean([r['onset_r'] for r in rows]),
		bar_f1=mean([r['bar_f1'] for r in rows]),
		rel_f1=mean([r['rel_f1'] for r in rows]),
		rel_p=mean([r['rel_p'] for r in rows]), rel_r=mean([r['rel_r'] for r in rows]),
		reon_rate=mean([r['reon_rate'] for r in rows]),
		shift_f1=mean([r['shift_f1'] for r in rows]),
		span_f1=mean([r['span_f1'] for r in rows]),
		stalls=mean([r['stalls'] for r in rows]),
		# How many files' best shift is explained by drift inherited from the SKIPPED bars. The
		# correcting shift opposes the offset, so an output already `anchor_delta` late at the window
		# start is realigned by `-anchor_delta`: the two sum to ~0 when the drift is inherited.
		inherited=sum(1 for r in rows if r['anchor_delta'] is not None
			and r['shift_ticks'] != 0 and abs(r['anchor_delta'] + r['shift_ticks']) <= args.grid),
		shifted_files=sum(1 for r in rows if r['shift_ticks'] != 0),
		stuck_out=mean([r['stuck_out'] for r in rows]),
		bars=mean([r['measures_scored'] for r in rows]),
		notes_out=mean([r['notes_out'] for r in rows]),
		notes_ref=mean([r['notes_ref'] for r in rows]),
		seconds=mean([r['seconds'] for r in rows]))
	return rows, agg


def main ():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--run', default=T.DEFAULT_RUN)
	ap.add_argument('--checkpoint', default=None,
		help='PIN THIS. best.chkpt in the mirror is overwritten hourly.')
	ap.add_argument('--corpus', required=True,
		help='dir holding midi-seq2-irregular/ and midi-seq2-score/ with matching names')
	ap.add_argument('--source-arm', default='midi-seq2-irregular')
	ap.add_argument('--target-arm', default='midi-seq2-score')
	ap.add_argument('--files', type=int, default=8, help='how many pairs (0 = all)')
	ap.add_argument('--skip', type=int, default=0, help='skip the first N pairs')
	ap.add_argument('--src-window', type=int, default=640)
	ap.add_argument('--sweep', default=None,
		help='comma-separated settings. Bare `960` sweeps src_window at --prime-window; '
			'`1260:520` pins both, so one paired run can vary either axis.')
	ap.add_argument('--advance-tokens', type=int, default=1,
		help='minimum target tokens retired per step, rounded to the next <eom> (default 1)')
	ap.add_argument('--prime-window', type=int, default=2048,
		help='internal target-view safety ceiling for sweep settings (default 2048)')
	ap.add_argument('--no-prime', action='store_true')
	ap.add_argument('--max-token', type=int, default=2048)
	ap.add_argument('--max-steps', type=int, default=12,
		help='windows per file (0 = whole file, slow)')
	ap.add_argument('--shift-span', type=int, default=3840,
		help='search a constant tick shift up to +/- this and report the best onset F1. Separates '
			'"wrong rhythm" from "right music at the wrong absolute tick"; 0 disables.')
	ap.add_argument('--grid', type=int, default=120,
		help='onset quantization in ticks; 120 = strict 16th at 480 tpb')
	ap.add_argument('--skip-bars', type=int, default=6,
		help='ignore this many opening bars. Step 0 generates a whole window of target in one shot, '
			'so bars from the start grade single-shot generation rather than the sliding seam where '
			'src_window matters.')
	ap.add_argument('--score-bars', type=int, default=8,
		help='score exactly this many bars per file (0 = whatever the output reached). Fixed by '
			'default because a larger src_window covers more music at the same --max-steps, so a '
			'sweep would otherwise compare different content.')
	ap.add_argument('--temperature', type=float, default=0.0)
	ap.add_argument('--top-k', type=int, default=0)
	ap.add_argument('--top-p', type=float, default=1.0)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--out', default=None, help='write the full result as JSON')
	args = ap.parse_args()

	src_dir = os.path.join(args.corpus, args.source_arm)
	ref_dir = os.path.join(args.corpus, args.target_arm)
	names = sorted(n for n in os.listdir(src_dir) if os.path.isfile(os.path.join(ref_dir, n)))
	if args.skip:
		names = names[args.skip:]
	if args.files:
		names = names[:args.files]
	pairs = [(n, os.path.join(src_dir, n), os.path.join(ref_dir, n)) for n in names]
	if not pairs:
		print('[error] no paired files found')
		return 1

	checkpoint = T.resolve_checkpoint(args.run, T.Configuration.createOrLoad(args.run,
		volatile=True), args.checkpoint)
	config, model = T.load_model(args.run, checkpoint, args.device)
	tokenizer, vocab_path = T.resolve_tokenizer(args.run, config)
	print(f'[vocab] {vocab_path} ({tokenizer.vocab_size} tokens)')
	print(f'[corpus] {len(pairs)} pairs from {args.corpus}')
	print(f'[grid] onsets quantized to {args.grid} ticks; max_steps {args.max_steps}')

	settings = (parse_settings(args.sweep, args.prime_window) if args.sweep
		else [(args.src_window, args.prime_window)])
	results = []
	for src_w, prime_w in settings:
		if args.no_prime:
			print(f'\n=== src_window {src_w}  NO PRIME (steady-state T {src_w + 1}) ===')
		else:
			print(f'\n=== src_window {src_w}  prime_window {prime_w}  '
				f'advance_tokens {args.advance_tokens}  (steady-state T {src_w + 1 + prime_w}) ===')
		rows, agg = run_setting(model, tokenizer, config, pairs, src_w, prime_w, args)
		if agg:
			print(f'  MEAN  pitchF1 {agg["pitch_f1"]:.3f} (p {agg["pitch_p"]:.3f} '
				f'r {agg["pitch_r"]:.3f})   onsetF1 {agg["onset_f1"]:.3f} '
				f'(p {agg["onset_p"]:.3f} r {agg["onset_r"]:.3f})   '
				f'barF1 {agg["bar_f1"]:.3f}   relF1 {agg["rel_f1"]:.3f}')
			results.append(dict(aggregate=agg, files=rows,
				names=[r['name'] for r in rows]))

	if len(results) > 1:
		# PAIRED comparison: keep only files every setting managed to score, so the windows are
		# compared on one common file set. Averaging over per-setting file sets would let an easy
		# file that only one window reached move that window's mean.
		common = set(results[0]['names'])
		for r in results[1:]:
			common &= set(r['names'])
		dropped = max(len(r['names']) for r in results) - len(common)
		print('\n=== sweep summary: %d files scored by every setting%s, bars %d..%d, grid %d ==='
			% (len(common), f' ({dropped} dropped as not common)' if dropped else '',
				args.skip_bars + 1, args.skip_bars + args.score_bars, args.grid))
		print('  %-12s %8s %8s %8s %8s %7s %7s %8s' % ('src:prime', 'pitchF1', 'onsetF1',
			'barF1', 'relF1', 're-on', 'notes', 'sec/file'))
		table = []
		for r in results:
			rows = [x for x in r['files'] if x['name'] in common]
			if not rows:
				continue
			a = dict(label=r['aggregate']['label'],
				pitch_f1=mean([x['pitch_f1'] for x in rows]),
				onset_f1=mean([x['onset_f1'] for x in rows]),
				bar_f1=mean([x['bar_f1'] for x in rows]),
				rel_f1=mean([x['rel_f1'] for x in rows]),
				reon_rate=mean([x['reon_rate'] for x in rows]),
				notes_out=mean([x['notes_out'] for x in rows]),
				seconds=mean([x['seconds'] for x in rows]))
			table.append(a)
			print('  %-12s %8.3f %8.3f %8.3f %8.3f %6.1f%% %7.1f %8.1f' % (a['label'],
				a['pitch_f1'], a['onset_f1'], a['bar_f1'], a['rel_f1'], a['reon_rate'] * 100,
				a['notes_out'], a['seconds']))
		if table:
			best = max(table, key=lambda a: a['onset_f1'])
			spread = max(a['onset_f1'] for a in table) - min(a['onset_f1'] for a in table)
			print(f'\n  best onset F1: {best["label"]} at {best["onset_f1"]:.3f}'
				f'   (spread across settings {spread:.3f})')
			# Per-file win counts, because a mean over few files can be carried by one outlier.
			wins = Counter()
			for name in sorted(common):
				per = [(a['label'], next(x['onset_f1'] for x in r['files']
					if x['name'] == name)) for a, r in zip(table, results)]
				wins[max(per, key=lambda t: t[1])[0]] += 1
			print('  per-file wins on onset F1: ' + ', '.join(
				f'{a["label"]}:{wins.get(a["label"], 0)}' for a in table))

	if args.out:
		with open(args.out, 'w', encoding='utf-8') as f:
			json.dump(dict(checkpoint=checkpoint, corpus=args.corpus, grid=args.grid,
				max_steps=args.max_steps, temperature=args.temperature, results=results), f,
				indent=2)
		print(f'[done] wrote {args.out}')
	return 0


if __name__ == '__main__':
	sys.exit(main())
