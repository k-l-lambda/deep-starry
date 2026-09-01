'''Align one generated midiseq2 file against its source and draw the correspondence.

Answers a question the accuracy harness cannot: not "how many onsets agree" but WHERE the generated
stream stops corresponding to its source, and how badly. `tests/midi/translate_accuracy_check.py`
scores a translation against ground truth by counting quantised onset matches, which yields one
number per file; this instead runs starry/midi/align.py over the pair and reports the alignment note
by note, so a defect has a location and a magnitude.

  GREEN triangle        the generated note matched a source note within --loss-threshold
  RED filled triangle   the generated note matched NO source note (align.py charged MissCost)
  RED hollow triangle   it matched, but its own self_cost cleared --loss-threshold
  link colour           that note's self_cost: grey-blue (0) through orange to deep red
  link width / alpha    inverse of the same cost, so a confident correspondence reads solid and a
                        strained one reads thin -- cost is on three channels, never colour alone

Layout follows plot_attention_step: two stacked panels, source above and generated below, one SHARED
pitch range so a same-pitch link comes out vertical, each lane's x normalised over its OWN tick
extent with its own tick labels (source along the top edge, generated along the bottom). So link
geometry is the diagnostic — vertical means the note is where it should be, tilt is how far off, and
different end heights mean the pitch disagrees.

Deliberately NOT the two-pin placement that plot_attention_step uses. Those pins exist to remove a
sliding step's carried-in offset, and pinning here would spend the alignment's own conclusion on
placing the axes and then draw links that are vertical by construction — the figure would assert what
it is supposed to be testing. Per-lane extent normalisation keeps the tilt honest.

The alignment always runs over the WHOLE pair; --from-measure/--to-measure slice only what is drawn.
Scoring a slice would restart the anchor mid-piece and hand the first notes of the view a cold start
they do not have in reality.

Run:  python tests/midi/align_match_viz.py                     (defaults to the I-YIgmEZ0ss pair)
      python tests/midi/align_match_viz.py --from-measure 4 --to-measure 10
      python tests/midi/align_match_viz.py --loss-threshold 0.3 --verbose
'''

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.midi.align import AlignState, Config, soft_indices
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer

sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'midi'))
from translateMidiseq2 import encode_lines, keyword_tokens, note_on_events


DEFAULT_SOURCE = os.path.expanduser('~/data/midi/test202606/midiseq2/I-YIgmEZ0ss.midiseq2.txt')
DEFAULT_TARGET = os.path.join(REPO_ROOT, 'tests/output/translate_midiseq2/I-YIgmEZ0ss.midiseq2.txt')
DEFAULT_OUT = os.path.join(REPO_ROOT, 'tests/output/align')

# Shared with plot_attention_step so the two figures read alike.
SRC_COLOR = '#2f5f9f'
OUT_COLOR = '#a5432f'		# generated lane's axis/frame identity (kept from plot_attention_step)
# Green-vs-red is the deutan/protan confusion axis, so this green is chosen by measurement, not by
# eye. Against MISS_COLOR: dE 30.2 normal, 11.5 protan, 10.8 deutan (OKLab x100) -- both dichromat
# margins clear 8 unaided. The two deficiencies pull opposite ways (protan wants the green lighter,
# deutan wants its lightness further from the red's), and this slightly teal-shifted green is the
# balance point; a plainer #2f8f4f measures 6.6 under deutan, which would need shape to be readable.
MATCH_COLOR = '#1f7a4d'	# a generated note that aligned
MISS_COLOR = '#d81f1f'
MIN_LINK_ALPHA = 0.12


def load_events (path, tokenizer, keywords, eom=False):
	'''One midiseq2 file -> (note_on events with softIndex, <eom> ticks).

	`eom` mirrors the feeder's source_eom: the generated arm carries @measure directives and the
	source does not, so only the generated side has bar lines to collect.
	'''
	lines = open(path).read().splitlines()
	ids = encode_lines(lines, tokenizer, eom)
	marks = [] if eom else None
	events, _tick, _state = note_on_events(ids, tokenizer, keywords, marks=marks)
	sis = soft_indices([e['onset'] for e in events])
	for e, si in zip(events, sis):
		e['softIndex'] = si
	return events, [tick for _index, tick in (marks or [])]


def measure_of (tick, eoms):
	'''Bar number for a tick, from the generated stream's own <eom> ticks.

	<eom> marks the OPENING of bar N for N >= 2 (@measure 1 emits no token), which is the same
	convention render_lines counts on: notes before the first mark are in bar 1.
	'''
	n = 1
	for mark in eoms:
		if tick < mark:
			break
		n += 1
	return n


def miss_distance (state, pitch, tgt_softindex, anchor):
	'''For an UNMATCHED note: how far the nearest same-pitch source note was, in offset units.

	A miss count on its own is ambiguous. "Pitch present in the source but outside the candidate
	window" is equally consistent with a real generation defect and with a window too tight for this
	pair -- and the two call for opposite responses. The distance separates them: a note whose nearest
	same-pitch source neighbour sits just past the span was refused by the window, while one whose
	nearest neighbour is bars away has no counterpart to find.

	Returns (distance, src_index) or (None, None) when the pitch is absent from the source entirely.
	'''
	hits = state.src_by_pitch.get(pitch)
	if not hits or anchor is None:
		return None, None
	best = min(hits, key=lambda i: abs((state.src_events[i]['softIndex'] - tgt_softindex) - anchor))
	return abs((state.src_events[best]['softIndex'] - tgt_softindex) - anchor), best


def align_pair (src_events, out_events, seed_offset=0.0):
	'''Run align.py over the pair. Returns (state, per-note records).

	`seed_offset` bounds the cold start. Both files are whole-piece translations of the same music
	starting at bar 1, so their first onsets are the same musical event and offset ~0 there is a fact
	about the data rather than an assumption — without it the first generated note may match any
	same-pitch note anywhere in the file, since whole-file alignment has none of the bounding a
	source window gives sliding inference.
	'''
	state = AlignState(src_events, seed_offset=seed_offset)
	records = []
	for e in out_events:
		# read the anchor BEFORE observing, so a miss is diagnosed against the same anchor the
		# candidate window actually used rather than one this note has already updated
		anchor, _conf, _lo, _hi = state.anchor(e['softIndex'])
		if anchor is None:
			anchor = state.seed_offset
		detail = state.observe(e['pitch'], e['onset'], e['softIndex'])
		if detail['src'] is None:
			dist, near = miss_distance(state, e['pitch'], e['softIndex'], anchor)
			detail = dict(detail, miss_distance=dist, nearest=near)
		records.append(dict(out=e, **detail))
	return state, records


def report (state, records, out_events, eoms, threshold, verbose=False):
	'''Print the alignment, and the bars whose notes align worst.'''
	misses = [r for r in records if r['src'] is None]
	matched = [r for r in records if r['src'] is not None]
	high = [r for r in matched if r['self_cost'] is not None and r['self_cost'] >= threshold]
	costs = sorted(r['self_cost'] for r in matched if r['self_cost'] is not None)

	def pct (p):
		return costs[min(len(costs) - 1, int(len(costs) * p))] if costs else float('nan')

	print(f'  generated note_on   {len(out_events)}')
	print(f'  matched             {len(matched)}  ({len(matched) / max(1, len(out_events)) * 100:.1f}%)')
	print(f'  unmatched (miss)    {len(misses)}')
	print(f'  self_cost >= {threshold:<6.3g} {len(high)}')
	if costs:
		print(f'  self_cost  median {pct(0.5):.4f}  p90 {pct(0.9):.4f}  p99 {pct(0.99):.4f}  '
			f'max {costs[-1]:.4f}')
	print(f'  cost {state.cost:.3f}   value {state.value:.1f}   prior {state.prior:+.3f}   '
		f'ratio {state.ratio if state.ratio is None else round(state.ratio, 4)}   '
		f'residual {state.residual if state.residual is None else round(state.residual, 1)}')

	# Per-bar attribution: a count with no location is not actionable, and the generated stream's own
	# <eom> ticks are the only barring either file agrees on.
	by_bar = {}
	for r in records:
		bar = measure_of(r['out']['onset'], eoms)
		slot = by_bar.setdefault(bar, [0, 0, 0])
		slot[0] += 1
		slot[1] += r['src'] is None
		slot[2] += r['src'] is not None and r['self_cost'] is not None and r['self_cost'] >= threshold
	worst = sorted(by_bar.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[:8]
	bad = [f'bar {b} {m + h}/{n}' for b, (n, m, h) in worst if m + h]
	print(f'  worst bars (miss+high / notes): {", ".join(bad) if bad else "none"}')

	# Why each miss missed. The span is the candidate window's own half-width, so "just outside" means
	# within one more span of it -- refused by the window rather than absent from the music.
	span = Config['AnchorSoftSpan']
	dists = [r.get('miss_distance') for r in misses]
	absent = sum(1 for d in dists if d is None)
	near = sum(1 for d in dists if d is not None and d <= 2 * span)
	far = sum(1 for d in dists if d is not None and d > 2 * span)
	if misses:
		known = sorted(d for d in dists if d is not None)
		print(f'  miss breakdown: {absent} pitch absent, {near} just outside window '
			f'(<= {2 * span:g} si), {far} far'
			+ (f'   nearest-neighbour distance median {known[len(known) // 2]:.2f} si, '
				f'max {known[-1]:.2f}' if known else ''))

	if verbose:
		for i, r in enumerate(records):
			bar = measure_of(r['out']['onset'], eoms)
			if r['src'] is None:
				d = r.get('miss_distance')
				why = ('pitch absent' if d is None
					else f'nearest same-pitch {d:.2f} si away'
						+ (' (just outside window)' if d <= 2 * Config['AnchorSoftSpan'] else ''))
				print(f'    #{i:4d} bar {bar:3d} pitch {r["out"]["pitch"]:3d} '
					f'tick {r["out"]["onset"]:7d}  MISS: {why}')
			elif r['self_cost'] >= threshold:
				s = r['src']
				print(f'    #{i:4d} bar {bar:3d} pitch {r["out"]["pitch"]:3d} '
					f'tick {r["out"]["onset"]:7d}  -> src #{s} tick '
					f'{state.src_events[s]["onset"]:7d}  self_cost {r["self_cost"]:.4f} '
					f'offset {r["offset"]:+.3f} skip {r["skip"]}')
	return dict(matched=len(matched), misses=len(misses), high=len(high),
		median=pct(0.5) if costs else None, p99=pct(0.99) if costs else None)


def plot (state, records, src_events, out_events, eoms, path, threshold, subtitle=''):
	'''Draw the correspondence. Style follows plot_attention_step; see the module docstring.'''
	import matplotlib
	matplotlib.use('Agg')					# file output only; no display on a training box
	import matplotlib.pyplot as plt
	from matplotlib.colors import LinearSegmentedColormap, Normalize
	from matplotlib.cm import ScalarMappable
	from matplotlib.patches import ConnectionPatch
	from matplotlib.ticker import MaxNLocator

	def extent (events):
		ons = [e['onset'] for e in events]
		return (min(ons), max(ons)) if ons else (0, 0)

	src_lo, src_hi = extent(src_events)
	out_lo, out_hi = extent(out_events)

	# Each lane over its OWN extent. Not the two-pin placement of plot_attention_step: those pins
	# remove a sliding step's inherited offset, and applying them here would place the axes using the
	# alignment's own conclusion, making every link vertical by construction.
	def norm (tick, lo, hi):
		return 0.5 if hi <= lo else (tick - lo) / (hi - lo)

	norm_s = lambda t: norm(t, src_lo, src_hi)
	norm_o = lambda t: norm(t, out_lo, out_hi)

	fig, (ax_s, ax_o) = plt.subplots(2, 1,
		figsize=(max(11, min(34, (len(src_events) + len(out_events)) / 7.0)), 6.8))

	# One shared pitch RANGE, each panel still ticked on its own axis: a same-pitch link must come out
	# vertical, and per-panel autoscaling would tilt it by whatever the two tessituras differ by.
	pitches = [e['pitch'] for e in src_events] + [e['pitch'] for e in out_events]
	p_lo, p_hi = (min(pitches), max(pitches)) if pitches else (60, 61)
	pad = max(2, (p_hi - p_lo) * 0.08)
	for ax in (ax_s, ax_o):
		ax.set_xlim(-0.03, 1.03)
		ax.set_ylim(p_lo - pad, p_hi + pad)
		ax.set_xticks([n * 0.1 for n in range(11)])
		ax.set_ylabel('pitch')
		ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
		ax.grid(alpha=0.18, zorder=1)
		# Links live on ax_s with clipping off so they reach into ax_o, which is drawn after; making
		# both patches transparent keeps the crossing segment visible over its whole length.
		ax.patch.set_visible(False)

	fracs = [n * 0.1 for n in range(11)]
	ax_s.xaxis.set_ticks_position('top')
	ax_s.xaxis.set_label_position('top')
	ax_s.set_xticklabels([f'{src_lo + f * (src_hi - src_lo):.0f}' for f in fracs])
	ax_s.set_xlabel(f'source onset (ticks {src_lo}..{src_hi}, normalised)', color=SRC_COLOR)
	ax_s.tick_params(axis='x', colors=SRC_COLOR)
	for side in ('top', 'left'):
		ax_s.spines[side].set_color(SRC_COLOR)
	ax_o.set_xticklabels([f'{out_lo + f * (out_hi - out_lo):.0f}' for f in fracs])
	ax_o.set_xlabel(f'generated onset (ticks {out_lo}..{out_hi}, normalised)', color=OUT_COLOR)
	ax_o.tick_params(axis='x', colors=OUT_COLOR)
	for side in ('bottom', 'left'):
		ax_o.spines[side].set_color(OUT_COLOR)

	# Source notes split by whether the alignment used them: an unused source note is a candidate
	# nothing claimed, which is a different statement from a note that was matched, and on a figure
	# about correspondence the difference is the point.
	used = {r['src'] for r in records if r['src'] is not None}
	for indices, style, label in (
			(sorted(set(range(len(src_events))) - used), dict(s=18, marker='o', facecolors='none',
				edgecolors=SRC_COLOR, linewidths=0.6, alpha=0.45), 'source, unmatched'),
			(sorted(used), dict(s=26, marker='o', c=SRC_COLOR, edgecolors='white',
				linewidths=0.4), 'source, matched')):
		if indices:
			ax_s.scatter([norm_s(src_events[i]['onset']) for i in indices],
				[src_events[i]['pitch'] for i in indices],
				label=f'{label} ({len(indices)})', zorder=4, **style)
	ax_s.legend(loc='upper right', fontsize=8, framealpha=0.9)

	# Loss colour scale. A custom ramp rather than a named one: it has to stay monotone in lightness
	# (so it survives greyscale) AND end on the same red the miss markers use, so "red means wrong"
	# is one claim on this figure instead of two.
	cmap = LinearSegmentedColormap.from_list('align_loss',
		['#8fa8c8', '#c9c07a', '#e08a3c', '#c22b1e', '#7d1410'])
	hi_cost = max([threshold] + [r['self_cost'] for r in records
		if r['self_cost'] is not None])
	cnorm = Normalize(vmin=0.0, vmax=hi_cost)

	# Generated notes in three classes, and they must not read alike. Shape and fill carry the
	# distinction as well as colour, so it survives greyscale and colour-blind viewing.
	good = [r for r in records if r['src'] is not None and r['self_cost'] < threshold]
	high = [r for r in records if r['src'] is not None and r['self_cost'] >= threshold]
	miss = [r for r in records if r['src'] is None]
	# Green for aligned, red for not. The lane's own OUT_COLOR is a terracotta that sits between the
	# two on hue, so using it for the matched class would leave three warm classes and make "did this
	# note align" a judgement about shade. OUT_COLOR stays on the axis frame, where it identifies the
	# lane rather than a verdict. Shape still carries the split (filled / hollow / half-tone), so the
	# green-vs-red reading is reinforced, not relied upon.
	if good:
		ax_o.scatter([norm_o(r['out']['onset']) for r in good], [r['out']['pitch'] for r in good],
			s=30, marker='v', c=MATCH_COLOR, edgecolors='white', linewidths=0.4,
			label=f'generated, aligned ({len(good)})', zorder=4)
	if high:
		ax_o.scatter([norm_o(r['out']['onset']) for r in high], [r['out']['pitch'] for r in high],
			s=52, marker='v', facecolors='none', edgecolors=MISS_COLOR, linewidths=1.5,
			label=f'self_cost >= {threshold:g} ({len(high)})', zorder=5)
	# Misses split by WHY. A single red symbol for both would merge two different claims: a note whose
	# nearest same-pitch source neighbour sits just past the candidate window (the window refused it)
	# and one with no counterpart anywhere near (nothing to match). Same red -- both are defects on
	# this figure -- but filled vs half-tone, so the split is legible without reading the log.
	span2 = 2 * Config['AnchorSoftSpan']
	miss_near = [r for r in miss if r.get('miss_distance') is not None
		and r['miss_distance'] <= span2]
	miss_far = [r for r in miss if r not in miss_near]
	if miss_near:
		ax_o.scatter([norm_o(r['out']['onset']) for r in miss_near],
			[r['out']['pitch'] for r in miss_near],
			s=52, marker='v', c=MISS_COLOR, alpha=0.45, edgecolors=MISS_COLOR, linewidths=0.8,
			label=f'unmatched, just outside window ({len(miss_near)})', zorder=6)
	if miss_far:
		ax_o.scatter([norm_o(r['out']['onset']) for r in miss_far],
			[r['out']['pitch'] for r in miss_far],
			s=56, marker='v', c=MISS_COLOR, edgecolors='white', linewidths=0.5,
			label=f'unmatched, no counterpart ({len(miss_far)})', zorder=6)

	# Bar lines of the GENERATED stream: they are the model's own barring, and they make a horizontal
	# displacement musically readable -- a link landing a bar late is a different error from one
	# landing a beat late, and on a bare onset axis both merely look shifted.
	drawn = 0
	for tick in sorted(eoms):
		x = norm_o(tick)
		if not (-0.03 <= x <= 1.03):
			continue
		ax_o.axvline(x, color='#8a7f5a', lw=0.7, alpha=0.55, zorder=1,
			label=(f'<eom> ({len(eoms)})' if drawn == 0 else None))
		drawn += 1
	ax_o.legend(loc='upper right', fontsize=8, framealpha=0.9)

	# Links, in DATA coordinates of the two axes so they cross the panel boundary.
	for r in records:
		if r['src'] is None:
			continue					# a miss has nothing to link TO; the red marker is the statement
		cost = r['self_cost']
		w = 1.0 - min(1.0, cost / hi_cost) if hi_cost else 1.0		# confidence, for width/alpha
		src = state.src_events[r['src']]
		link = ConnectionPatch(
			xyA=(norm_s(src['onset']), src['pitch']), coordsA=ax_s.transData,
			xyB=(norm_o(r['out']['onset']), r['out']['pitch']), coordsB=ax_o.transData,
			color=cmap(cnorm(cost)),
			lw=0.4 + 1.5 * w,
			# Floored: a link drawn at 0.02 is one that was recorded and then hidden, which is the
			# worst of both -- the legend counts it and the figure does not show it.
			alpha=max(MIN_LINK_ALPHA, 0.18 + 0.62 * w),
			linestyle='-' if cost < threshold else (0, (4, 2)), zorder=2)
		link.set_clip_on(False)
		ax_s.add_artist(link)

	costs = [r['self_cost'] for r in records if r['self_cost'] is not None]
	med = sorted(costs)[len(costs) // 2] if costs else float('nan')
	fig.suptitle(f'{len(out_events)} generated vs {len(src_events)} source note_on   '
		f'{len(good)} aligned / {len(high)} high-cost / {len(miss)} unmatched   '
		f'median self_cost {med:.4f}   cost {state.cost:.3f} prior {state.prior:+.3f}'
		f'{subtitle}', fontsize=10)
	# ConnectionPatch resolves endpoints from transData at DRAW time, so laying out after adding the
	# links is safe: the segments follow the panels wherever the layout puts them.
	# Layout FIRST, then the colorbar into an axes of its own. A colorbar attached with ax=(ax_s, ax_o)
	# produces an axes tight_layout declares incompatible ("results might be incorrect"), and a layout
	# warning is not worth shipping.
	fig.tight_layout(rect=(0, 0.01, 0.94, 0.93))
	fig.subplots_adjust(hspace=0.34)
	cax = fig.add_axes((0.955, 0.12, 0.011, 0.7))
	bar = fig.colorbar(ScalarMappable(norm=cnorm, cmap=cmap), cax=cax)
	bar.set_label('per-note self_cost (offset inconsistency)', fontsize=8)
	bar.ax.tick_params(labelsize=7)
	fig.savefig(path, dpi=130)
	plt.close(fig)
	return path


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--source', default=DEFAULT_SOURCE, help='source (irregular arm) midiseq2')
	ap.add_argument('--target', default=DEFAULT_TARGET, help='generated (score arm) midiseq2')
	ap.add_argument('--out-dir', default=DEFAULT_OUT)
	ap.add_argument('--loss-threshold', type=float, default=0.5,
		help='self_cost at or above which a matched note is drawn red (dashed link)')
	ap.add_argument('--seed-offset', type=float, default=0.0,
		help='cold-start offset prior; use "nan" to disable and let the first note match anywhere')
	ap.add_argument('--from-measure', type=int, default=0, help='first generated bar to DRAW')
	ap.add_argument('--to-measure', type=int, default=0, help='last generated bar to DRAW')
	ap.add_argument('--anchor-span', type=float, default=None,
		help=f'candidate window half-width in softIndex (default {Config["AnchorSoftSpan"]}). '
			'The two arms advance softIndex at different rates when one collapses events into '
			'chords, so this is the knob that decides miss vs match')
	ap.add_argument('--verbose', action='store_true', help='list every miss and high-cost note')
	args = ap.parse_args()

	for path in (args.source, args.target):
		if not os.path.isfile(path):
			print(f'not found: {path}')
			return 1

	if args.anchor_span is not None:
		Config['AnchorSoftSpan'] = args.anchor_span

	tk = Midiseq2Tokenizer()
	kw = keyword_tokens(tk)
	src_events, _ = load_events(args.source, tk, kw, eom=False)
	out_events, eoms = load_events(args.target, tk, kw, eom=True)
	print(f'source {os.path.basename(args.source)}: {len(src_events)} note_on')
	print(f'target {os.path.basename(args.target)}: {len(out_events)} note_on, {len(eoms)} <eom>')

	seed = None if args.seed_offset != args.seed_offset else args.seed_offset	# nan -> disabled
	state, records = align_pair(src_events, out_events, seed_offset=seed)
	stats = report(state, records, out_events, eoms, args.loss_threshold, args.verbose)

	# Slice only what is DRAWN. The alignment above ran over the whole pair on purpose: scoring a
	# slice would restart the anchor mid-piece and hand the view's first notes a cold start they do
	# not have in reality.
	draw_records, draw_out = records, out_events
	subtitle = ''
	if args.from_measure or args.to_measure:
		lo = args.from_measure or 1
		hi = args.to_measure or 10 ** 9
		draw_records = [r for r in records if lo <= measure_of(r['out']['onset'], eoms) <= hi]
		draw_out = [r['out'] for r in draw_records]
		if not draw_out:
			print(f'no generated notes in bars {lo}..{hi}')
			return 1
		subtitle = f'   bars {lo}..{min(hi, measure_of(draw_out[-1]["onset"], eoms))}'

	# The source side is drawn over the span the DRAWN notes actually matched, padded, rather than
	# the whole file: with a slice selected, plotting every source note would compress the matched
	# region to a sliver and the tilt -- the whole point of the geometry -- would stop being readable.
	used = [r['src'] for r in draw_records if r['src'] is not None]
	if used and (args.from_measure or args.to_measure):
		pad = max(4, (max(used) - min(used)) // 8)
		draw_src = src_events[max(0, min(used) - pad):max(used) + pad + 1]
	else:
		draw_src = src_events
	eom_draw = [t for t in eoms
		if draw_out and draw_out[0]['onset'] <= t <= draw_out[-1]['onset']] if draw_out else []

	os.makedirs(args.out_dir, exist_ok=True)
	stem = os.path.basename(args.target).replace('.midiseq2.txt', '').replace('.txt', '')
	span = f'.bars{args.from_measure or 1}-{args.to_measure}' if (args.from_measure or args.to_measure) else ''
	path = os.path.join(args.out_dir, f'{stem}.align{span}.png')
	plot(state, draw_records, draw_src, draw_out, eom_draw, path, args.loss_threshold, subtitle)
	print(f'\nwrote {path}  ({len(draw_out)} generated, {len(draw_src)} source drawn)')
	return 0


if __name__ == '__main__':
	sys.exit(main())
