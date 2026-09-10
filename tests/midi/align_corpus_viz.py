'''Draw one corpus pair's alignment WITH the ground truth on the same figure.

`tests/midi/align_corpus_check.py` scores 100 pairs and prints aggregate numbers; a number cannot say
WHICH note went wrong. `tests/midi/align_match_viz.py` draws a correspondence note by note, but it
draws a TRANSLATION against its source, where no ground truth exists -- so it can only colour a link
by align.py's own `self_cost`, i.e. by the aligner's opinion of itself.

This file exists because a corpus pair has something neither of those has: `@tick` ADJUDICATES every
link. The score and irregular arms are the same piece, and both carry the same `(measure, tick)`
directives while their real onsets diverge (score quantised 0,240,480...; irregular 0,201,411... on one
bar), so for each generated note the group it BELONGS to is known independently of the aligner. That
turns the figure from "how confident was the aligner" into "was it right", and those are different
pictures -- a link can be confident and wrong, which is the case worth seeing and the one `self_cost`
colouring cannot show.

What is drawn, per generated (score-arm) note:

  GREEN link       align.py matched a source note in the CORRECT (measure, tick) group
  RED link         it matched, but the source note is in the WRONG group -- and a GREY reference link
                   is drawn alongside it, to where the truth says it should have pointed
  RED hollow mark  it matched nothing at all (align.py charged MissCost)
  GREY-RING mark   no shared key, so this note is UNADJUDICABLE -- neither credited nor blamed

The grey truth-link is the whole point of the figure: on a wrong match you see BOTH where the aligner
went and where it should have gone, so the error has a direction and a magnitude rather than just a
colour. Where the two links coincide the match is right, so a correct alignment reads as green links
with no grey visible.

Colour is validated, not chosen by eye. MATCH/MISS are reused unchanged from `align_match_viz.py`
(measured dE 30.2 normal / 11.4 protan / 10.7 deutan in OKLab x100 -- recomputed here and it
reproduces that file's documented figures). The truth-grey was then MEASURED against both: #7a7a7a,
the obvious choice, sits at 12.7 normal against MATCH and FAILS the 15 floor, so it would have been a
third category the eye cannot separate from the verdict. #bdbdbd measures 30.4/25.6/28.7 against MATCH
and 31.8/37.0/25.6 against MISS, clearing the normal floor and both CVD margins. Verdict is on shape
and link style as well as hue, so none of it is colour-alone.

Layout follows `align_match_viz.py` (two stacked lanes, source above, one SHARED pitch range so a
same-pitch link is vertical, each lane normalised over its OWN tick extent) for one reason: the two
figures are read side by side, and a reader should not have to relearn the geometry. Per-lane
normalisation over a two-pin placement for that file's reason too -- pinning would spend the
alignment's own conclusion on placing the axes and make links vertical by construction.

The alignment always runs over the WHOLE window; --from-measure/--to-measure slice only what is DRAWN.
Scoring a slice would restart the anchor mid-piece and hand the first drawn notes a cold start they do
not have in reality.

Run:  python tests/midi/align_corpus_viz.py                          (first pair, by sorted name)
      python tests/midi/align_corpus_viz.py --sample 168b599d        (name or unique prefix)
      python tests/midi/align_corpus_viz.py --sample 14d850fc --from-measure 2 --to-measure 6
      python tests/midi/align_corpus_viz.py --worst 3                (the 3 worst-aligned pairs)
'''

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.midi.align import AlignState

sys.path.insert(0, os.path.join(REPO_ROOT, 'tests', 'midi'))
# Reuse the check's loaders rather than re-deriving them. The figure must be about the SAME alignment
# the check scores -- a second copy of the key/measure attribution could drift from it, and then a
# green link would not mean the check counted a hit.
from align_corpus_check import (SRC_ARM, TGT_ARM, SRC_WINDOW, load_notes, note_keys, pair_files,
	resolve_dir)
from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer

sys.path.insert(0, os.path.join(REPO_ROOT, 'tools', 'midi'))
from translateMidiseq2 import keyword_tokens


DEFAULT_ROOT = os.path.expanduser('~/data/midi/test202608')
DEFAULT_OUT = os.path.join(REPO_ROOT, 'tests/output/align_corpus')

# Lane identity, shared with align_match_viz.py so the two figures read alike.
SRC_COLOR = '#2f5f9f'
OUT_COLOR = '#a5432f'
# Verdict axis, reused unchanged from align_match_viz.py (validated there, re-measured in the header).
MATCH_COLOR = '#1f7a4d'
MISS_COLOR = '#d81f1f'
# The ground-truth reference link. MEASURED (see header): the obvious #7a7a7a fails the normal-vision
# floor against MATCH_COLOR at 12.7, this clears it at 30.4 and holds >= 25 on both CVD axes.
TRUTH_COLOR = '#bdbdbd'
# A note the ground truth cannot judge. Deliberately neither verdict colour and neither the truth grey:
# it is an absence of evidence, and painting it as a verdict would be a claim the data does not make.
UNJUDGED_COLOR = '#6f6f6f'
# A match that is the right NOTE but a key the mocker moved. Must read as neither verdict: it is not an
# error, and painting it green would assert the keys agreed.
#
# CHOSEN BY MEASUREMENT, after the obvious blue failed. #3f7fa8 measures only 14.5 normal against
# MATCH -- under the 15 floor -- and 9.6 against BOTH the unjudged grey and the source lane's own blue,
# so it would have collided with three things at once. This violet clears every pair it has to
# (OKLab dE x100, normal/protan/deutan):
#     vs MATCH 29.9/21.0/21.2   vs MISS 25.4/28.1/25.8   vs TRUTH-grey 21.6/22.7/18.5
#     vs UNJUDGED 19.7/16.1/16.5   vs SRC lane 21.6/12.2/17.2
# It has to clear five, not two, because this figure already spends green/red on the verdict, grey on
# the truth reference and blue on the source lane -- a fifth class has nowhere obvious left to go, and
# violet is the one direction not already claimed. Shape (diamond) carries it as well, so no reading
# depends on hue alone.
ROLL_COLOR = '#b070d8'


def build_pair (src_dir, tgt_dir, name, src_window):
	'''(src, tgt) note lists for one piece, with `key` and `measure` attached. Via the check's loader.'''
	return pair_files(src_dir, tgt_dir, name, src_window)


def group_source (src):
	'''(measure, tick) -> the set of source indices in that group.

	A GROUP, not one index: a chord shares a key and its notes may be paired in any order, so a
	correct match is membership. Demanding a specific index would paint the chords red.
	'''
	out = {}
	for i, e in enumerate(src):
		key = e.get('key')
		if key and key[1] is not None:
			out.setdefault(key, set()).add(i)
	return out


def rolled_groups (src, tgt):
	"""(bar, pitch) -> source indices the mocker MOVED out of their score group.

	The mocker rolls chords: the score's (1,480)[57,60,65,69,72] becomes (1,480)[57,60] (1,543)[65]
	(1,607)[69] (1,672)[72] in the irregular arm. Those keys are not quantised (the irregular arm holds
	597 non-multiple-of-10 @tick values against the score arm's 0) and MEASURED corpus-wide 90.6% of
	the notes in an irregular-only key are the same pitch in the same bar as a score note -- moved, not
	added. Counting those as errors would charge align.py for finding the musically right note.

	Restricted to keys the SCORE ARM DOES NOT HAVE, which is what stops this from being a licence: such
	a key is no other target note's truth, so admitting it creates no ambiguity about ownership.
	"""
	tgt_keys = {e['key'] for e in tgt if e.get('key') and e['key'][1] is not None}
	out = {}
	for i, e in enumerate(src):
		key = e.get('key')
		if key and key[1] is not None and key not in tgt_keys:
			out.setdefault((key[0], e['pitch']), set()).add(i)
	return out


def align_and_judge (src, tgt):
	'''Run align.py over the pair, adjudicating each match against the (measure, tick) truth.

	Returns (state, records). Each record carries the verdict AND, when the match is wrong, the source
	indices the truth points at -- which is what the grey reference link is drawn from. Without those
	the figure could say "wrong" but not "wrong by how much", and the magnitude is most of the
	diagnostic value.
	'''
	groups = group_source(src)
	rolled = rolled_groups(src, tgt)
	state = AlignState(src, seed_offset=0.0)
	records = []
	for j, e in enumerate(tgt):
		key = e.get('key')
		truth = groups.get(key) if (key and key[1] is not None) else None
		detail = state.observe(e['pitch'], e['onset'], e['softIndex'])
		index = detail['src']
		if index is None:
			verdict = 'miss'
		elif truth is None:
			verdict = 'unjudged'		# no shared key: neither credited nor blamed
		elif index in truth:
			verdict = 'right'
		elif index in rolled.get((key[0], e['pitch']), ()):
			# the same pitch, same bar, in a group the mocker created by rolling a chord. Drawn as its
			# own class rather than folded into 'right': it IS the right note, but the figure should
			# not silently claim the keys agreed when they did not.
			verdict = 'rolled'
		else:
			verdict = 'wrong'
		records.append(dict(tgt_index=j, src=index, verdict=verdict, key=key,
			truth=sorted(truth) if truth else None,
			self_cost=detail['self_cost'], cost=detail['cost'], offset=detail['offset']))
	return state, records


def summarise (records):
	'''Counts per verdict, plus the two rates the check reports, so figure and check are comparable.'''
	tally = dict(right=0, rolled=0, wrong=0, miss=0, unjudged=0)
	for r in records:
		tally[r['verdict']] += 1
	judged = tally['right'] + tally['rolled'] + tally['wrong']
	scorable = judged + tally['miss']		# a miss on a keyed note is a recall failure
	return dict(tally=tally,
		precision=tally['right'] / judged if judged else None,
		rolled_precision=(tally['right'] + tally['rolled']) / judged if judged else None,
		recall=tally['right'] / scorable if scorable else None)


def plot (src, tgt, records, stats, path, title, from_measure=None, to_measure=None,
		measures=None):
	'''Draw the correspondence with the truth overlaid. Geometry follows align_match_viz.plot.'''
	import matplotlib
	matplotlib.use('Agg')					# file output only; no display on a training box
	import matplotlib.pyplot as plt
	from matplotlib.patches import ConnectionPatch
	from matplotlib.ticker import MaxNLocator
	from matplotlib.lines import Line2D

	# Slice what is DRAWN, never what was scored -- see the module docstring.
	lo_m = from_measure or min(e['measure'] for e in tgt)
	hi_m = to_measure or max(e['measure'] for e in tgt)
	shown = [r for r in records if lo_m <= tgt[r['tgt_index']]['measure'] <= hi_m]
	if not shown:
		print(f'  nothing to draw in measures {lo_m}..{hi_m}')
		return False
	# A bar COUNT, which is the unit the music and the ground truth are both organised in. Counted from
	# the window's own first bar rather than from bar 1 of the piece: a 960-token window does not
	# necessarily start at the beginning (MEASURED: of the three weakest pairs, 04623a3b's window opens
	# at bar 2), so "the first 8 bars" is only well defined relative to what is in the window. The
	# absolute range actually drawn goes in the suptitle and the report so the two never get confused.
	#
	# Crops what is DRAWN, never what was SCORED: the rates in the suptitle and the text report stay
	# those of the whole window, so a cropped figure cannot flatter the alignment.
	cropped = 0
	if measures:
		bars = sorted({tgt[r['tgt_index']]['measure'] for r in shown})[:measures]
		keep = [r for r in shown if tgt[r['tgt_index']]['measure'] <= bars[-1]]
		cropped = len(shown) - len(keep)
		shown = keep
		lo_m, hi_m = bars[0], bars[-1]
		print(f'  drawing bars {lo_m}..{hi_m} ({len(bars)} of '
			f'{len({tgt[r["tgt_index"]]["measure"] for r in records})} in window), '
			f'{len(shown)} of {len(shown) + cropped} target onsets')
	# Source notes in view: those any drawn link touches, plus the truth groups of the wrong ones, so a
	# grey link always has its endpoint on the figure.
	touched = {r['src'] for r in shown if r['src'] is not None}
	for r in shown:
		if r['verdict'] == 'wrong' and r['truth']:
			touched.update(r['truth'])
	src_view = sorted(touched) or list(range(len(src)))
	tgt_view = [r['tgt_index'] for r in shown]

	def extent (onsets):
		return (min(onsets), max(onsets)) if onsets else (0, 0)

	src_lo, src_hi = extent([src[i]['onset'] for i in src_view])
	tgt_lo, tgt_hi = extent([tgt[j]['onset'] for j in tgt_view])

	def norm (tick, lo, hi):
		return 0.5 if hi <= lo else (tick - lo) / (hi - lo)

	norm_s = lambda t: norm(t, src_lo, src_hi)
	norm_t = lambda t: norm(t, tgt_lo, tgt_hi)

	# GEOMETRY IS THE DIAGNOSTIC, so the aspect ratio is a correctness concern rather than taste. The
	# width formula inherited from align_match_viz.py was sized for whole-file translations of many
	# hundreds of notes; on a 960-token window (~130 notes over 7-10 bars) it produced a 4.9:1 figure in
	# which every link was within a few degrees of horizontal -- so tilt, which the docstring calls the
	# diagnostic, could not be read at all, and even a perfect alignment came out a horizontal smear.
	# Capped near 2:1 instead: link tilt scales with the panel gap over the horizontal span, so bounding
	# the ratio is what keeps a wrong match visibly slanted.
	width = max(10.0, min(19.0, 5.0 + (len(src_view) + len(tgt_view)) / 14.0))
	fig, (ax_s, ax_t) = plt.subplots(2, 1, figsize=(width, max(7.5, width / 2.1)))

	pitches = [src[i]['pitch'] for i in src_view] + [tgt[j]['pitch'] for j in tgt_view]
	p_lo, p_hi = (min(pitches), max(pitches)) if pitches else (60, 61)
	pad = max(2, (p_hi - p_lo) * 0.08)
	for ax in (ax_s, ax_t):
		ax.set_xlim(-0.03, 1.03)
		ax.set_ylim(p_lo - pad, p_hi + pad)
		ax.set_ylabel('pitch')
		ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
		ax.grid(axis='y', alpha=0.18, zorder=1)
		# Links live on ax_s with clipping off so they reach into ax_t, drawn after; transparent
		# patches keep the crossing segment visible over its whole length.
		ax.patch.set_visible(False)

	# The x axis carries the (measure, tick) KEY, not a decimal onset fraction. That fraction was a
	# rendering coordinate nothing in this figure is judged in: every verdict above is `index in truth`
	# where truth is keyed by (measure, tick), so the axis should show the coordinate the verdict is
	# computed in or the reader cannot check the verdict against the picture.
	#
	# @tick is MEASURE-RELATIVE -- it resets at every @measure line -- so a tick alone is ambiguous and
	# only the pair identifies a group. Hence bar numbers as the labelled anchors (bold, `|N`) with the
	# in-bar tick values between them, rather than one flat running number.
	#
	# Each lane is gridded from ITS OWN notes, so a bar line sits at different x in the two lanes. That
	# offset is not an artifact to correct -- it IS the rubato: the source lane is the mocker's warped
	# time, the target lane is quantised, and the horizontal drift between the two grids is the thing
	# align.py has to undo. A shared grid would hide exactly that.
	def lane_grid (notes, view, norm_fn):
		'''[(x, x_end, measure, tick, is_barline)] per distinct KEY in view, left to right.

		One line per KEY, not per onset. MEASURED on 14d850fc: the irregular arm has 33 distinct keys
		across 119 distinct (onset, key) pairs -- 27 keys sit at several different real onsets, because
		the mocker spreads one quantised chord over several times (the score arm is 24 keys at 24
		onsets, exactly 1:1). Gridding per onset therefore printed the same tick number twice at two
		different x, which reads as a bug in the axis rather than as what it is. So the line goes at the
		key's FIRST onset and its spread is returned as x_end, drawn as a band: the rubato becomes a
		visible width instead of a duplicated label.
		'''
		spans = {}
		for i in view:
			key = notes[i].get('key')
			if not key or key[1] is None:
				continue
			lo, hi = spans.get(key, (None, None))
			o = notes[i]['onset']
			spans[key] = (o if lo is None else min(lo, o), o if hi is None else max(hi, o))
		seen = set()
		out = []
		for key in sorted(spans, key=lambda k: (spans[k][0], k)):
			m, tick = key
			first = m not in seen
			seen.add(m)
			lo, hi = spans[key]
			out.append((norm_fn(lo), norm_fn(hi), m, tick, first))
		return out

	def draw_grid (ax, notes, view, norm_fn, color):
		grid = lane_grid(notes, view, norm_fn)
		if not grid:
			ax.set_xticks([n * 0.1 for n in range(11)])
			return grid
		# Thin the LABELS by available pixels, never the lines: a dropped line would move an unlabelled
		# group into the wrong bar visually, whereas a dropped label costs only precision of reading.
		px = max(1.0, fig.get_size_inches()[0] * fig.dpi * 0.92)
		need = 46.0 / px					# a `|12`/`480` label needs roughly this much axis width
		labelled, last = [], -1e9
		for x, _xe, m, _tick, bar in grid:	# bar lines claim their slot first: they are the anchors
			if bar and x - last >= need * 0.62:
				labelled.append((x, f'|{m}', True)); last = x
		for x, _xe, _m, tick, bar in grid:
			if bar:
				continue
			if all(abs(x - lx) >= need for lx, _l, _b in labelled):
				labelled.append((x, str(tick), False))
		labelled.sort()
		ax.set_xticks([x for x, _l, _b in labelled])
		ax.set_xticklabels([l for _x, l, _b in labelled])
		for lbl, (_x, _l, is_bar) in zip(ax.get_xticklabels(), labelled):
			if is_bar:
				lbl.set_fontweight('bold')
				lbl.set_fontsize(9)
			else:
				lbl.set_fontsize(7)
				lbl.set_alpha(0.75)
		ax.set_xticks([x for x, _xe, _m, _t, _b in grid], minor=True)
		# Bar lines span the lane; group lines are faint. Both under the marks (zorder 1). Where a key
		# occupies a range of real onsets, that range is shaded -- the mocker's spread of one chord.
		for x, x_end, _m, _tick, bar in grid:
			if bar:
				ax.axvline(x, color=color, alpha=0.38, linewidth=1.1, zorder=1)
			else:
				ax.axvline(x, color='#000000', alpha=0.07, linewidth=0.6, zorder=1)
			if x_end - x > 0.002:
				ax.axvspan(x, x_end, color=color, alpha=0.07, linewidth=0, zorder=1)
		return grid

	ax_s.xaxis.set_ticks_position('top')
	ax_s.xaxis.set_label_position('top')
	src_grid = draw_grid(ax_s, src, src_view, norm_s, SRC_COLOR)
	ax_s.set_xlabel(f'{SRC_ARM} — the SOURCE, with rubato.  x = @measure (bold |N) / @tick, '
		f'onset ticks {src_lo}..{src_hi}', color=SRC_COLOR)
	ax_s.tick_params(axis='x', colors=SRC_COLOR)
	for side in ('top', 'left'):
		ax_s.spines[side].set_color(SRC_COLOR)
	tgt_grid = draw_grid(ax_t, tgt, tgt_view, norm_t, OUT_COLOR)
	ax_t.set_xlabel(f'{TGT_ARM} — the TARGET, quantised.  x = @measure (bold |N) / @tick, '
		f'onset ticks {tgt_lo}..{tgt_hi}', color=OUT_COLOR)
	ax_t.tick_params(axis='x', colors=OUT_COLOR)
	for side in ('bottom', 'left'):
		ax_t.spines[side].set_color(OUT_COLOR)

	# Source lane: split by whether the alignment claimed the note. An unclaimed source note is a
	# candidate nothing took, which on a figure about correspondence is a different statement from a
	# matched one.
	used = {r['src'] for r in shown if r['src'] is not None}
	unused = [i for i in src_view if i not in used]
	if unused:
		ax_s.scatter([norm_s(src[i]['onset']) for i in unused], [src[i]['pitch'] for i in unused],
			s=18, marker='o', facecolors='none', edgecolors=SRC_COLOR, linewidths=0.6, alpha=0.45,
			label=f'source, unclaimed ({len(unused)})', zorder=4)
	claimed = [i for i in src_view if i in used]
	if claimed:
		ax_s.scatter([norm_s(src[i]['onset']) for i in claimed], [src[i]['pitch'] for i in claimed],
			s=26, marker='o', c=SRC_COLOR, edgecolors='white', linewidths=0.4,
			label=f'source, claimed ({len(claimed)})', zorder=5)
	# `best` rather than a pinned corner: which corner is free depends on the piece's tessitura, and a
	# hardcoded one hides links on whichever files put notes there. MEASURED the hard way -- upper right
	# covered the source lane's own notes on the first pair drawn.
	ax_s.legend(loc='best', fontsize=8, framealpha=0.92)

	# TRUTH links first, so a verdict link is drawn OVER its reference and the two are visibly paired
	# rather than competing. Only for wrong matches: where the match is right the truth link would sit
	# under the green one and add nothing but ink.
	drew_truth = False
	for r in shown:
		if r['verdict'] != 'wrong' or not r['truth']:
			continue
		e = tgt[r['tgt_index']]
		# the truth group member nearest in pitch, so the reference link is the most charitable reading
		# of where the aligner should have gone rather than an arbitrary member of the chord
		best = min(r['truth'], key=lambda i: (abs(src[i]['pitch'] - e['pitch']), i))
		ax_s.add_artist(ConnectionPatch(
			xyA=(norm_t(e['onset']), e['pitch']), coordsA=ax_t.transData,
			xyB=(norm_s(src[best]['onset']), src[best]['pitch']), coordsB=ax_s.transData,
			color=TRUTH_COLOR, linewidth=2.6, alpha=0.85, zorder=2, linestyle=(0, (5, 2))))
		drew_truth = True

	# Verdict links.
	for r in shown:
		if r['src'] is None:
			continue
		e = tgt[r['tgt_index']]
		if r['verdict'] == 'right':
			color, width, alpha = MATCH_COLOR, 1.0, 0.55
		elif r['verdict'] == 'rolled':
			color, width, alpha = ROLL_COLOR, 1.0, 0.6
		elif r['verdict'] == 'wrong':
			color, width, alpha = MISS_COLOR, 1.4, 0.9
		else:
			color, width, alpha = UNJUDGED_COLOR, 0.7, 0.3
		ax_s.add_artist(ConnectionPatch(
			xyA=(norm_t(e['onset']), e['pitch']), coordsA=ax_t.transData,
			xyB=(norm_s(src[r['src']]['onset']), src[r['src']]['pitch']), coordsB=ax_s.transData,
			color=color, linewidth=width, alpha=alpha, zorder=3))

	# Target lane, four classes. Shape and fill carry the verdict as well as hue, so the reading
	# survives greyscale and colour-blind viewing -- the green/red pair is the CVD confusion axis and
	# must never be the only channel.
	classes = (
		('right', dict(marker='^', s=34, c=MATCH_COLOR, edgecolors='white', linewidths=0.4),
			'matched, correct group'),
		('rolled', dict(marker='D', s=26, c=ROLL_COLOR, edgecolors='white', linewidths=0.4),
			'right note, chord rolled by mocker'),
		('wrong', dict(marker='v', s=44, c=MISS_COLOR, edgecolors='white', linewidths=0.5),
			'matched, WRONG group'),
		('miss', dict(marker='v', s=40, facecolors='none', edgecolors=MISS_COLOR, linewidths=1.1),
			'no match at all'),
		('unjudged', dict(marker='o', s=20, facecolors='none', edgecolors=UNJUDGED_COLOR,
			linewidths=0.7, alpha=0.55), 'no shared key (unjudged)'))
	for verdict, style, label in classes:
		sel = [r for r in shown if r['verdict'] == verdict]
		if sel:
			ax_t.scatter([norm_t(tgt[r['tgt_index']]['onset']) for r in sel],
				[tgt[r['tgt_index']]['pitch'] for r in sel],
				label=f'{label} ({len(sel)})', zorder=6, **style)
	handles, labels = ax_t.get_legend_handles_labels()
	if drew_truth:
		handles.append(Line2D([0], [0], color=TRUTH_COLOR, linewidth=2.6, linestyle=(0, (5, 2))))
		labels.append('ground truth (@tick), where it differs')
	ax_t.legend(handles, labels, loc='best', fontsize=8, framealpha=0.92, ncol=2)

	t = stats['tally']
	prec = f"{stats['precision']:.3f}" if stats['precision'] is not None else 'n/a'
	rec = f"{stats['recall']:.3f}" if stats['recall'] is not None else 'n/a'
	roll = (f"{stats['rolled_precision']:.3f}" if stats['rolled_precision'] is not None else 'n/a')
	fig.suptitle(f'{title}\n@tick precision {prec} (rolled-tolerant {roll})  recall {rec}   '
		f'right {t["right"]}  rolled {t["rolled"]}  wrong {t["wrong"]}  miss {t["miss"]}  '
		f'unjudged {t["unjudged"]}   '
		f'(bars {lo_m}..{hi_m} of {max(e["measure"] for e in tgt)}'
		+ (f'; DRAWN bars {lo_m}..{hi_m}, {len(shown)} of {len(shown) + cropped} target onsets, '
			'rates are the full window' if cropped else '') + ')', fontsize=9)
	# subplots_adjust, NOT tight_layout: the links are ConnectionPatches spanning both axes with clipping
	# off, which tight_layout cannot measure -- it warns "Axes that are not compatible" and may move the
	# panels out from under the links it could not see. Fixed margins keep the two lanes where the
	# ConnectionPatch coordinates expect them.
	fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.09, hspace=0.30)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return True


def report (src, tgt, records, stats, verbose=False):
	'''Text summary. Printed as well as drawn, so a headless run is still informative.'''
	t = stats['tally']
	prec = f"{stats['precision']:.4f}" if stats['precision'] is not None else 'n/a'
	rec = f"{stats['recall']:.4f}" if stats['recall'] is not None else 'n/a'
	print(f'  {len(src)} source note_on, {len(tgt)} target note_on over '
		f'{max(e["measure"] for e in tgt)} bars')
	roll = (f"{stats['rolled_precision']:.4f}" if stats['rolled_precision'] is not None else 'n/a')
	print(f'  @tick verdict: right {t["right"]}, rolled {t["rolled"]}, wrong {t["wrong"]}, '
		f'miss {t["miss"]}, unjudged {t["unjudged"]}')
	print(f'  precision {prec} (rolled-tolerant {roll}), recall {rec}')
	if not verbose:
		return
	wrong = [r for r in records if r['verdict'] == 'wrong']
	if not wrong:
		print('  no wrong matches to list')
		return
	print(f'  the {min(len(wrong), 12)} wrong matches (of {len(wrong)}):')
	print('    bar  tick  pitch   matched src  its key        truth key')
	for r in wrong[:12]:
		e = tgt[r['tgt_index']]
		s = src[r['src']]
		print(f'    {e["measure"]:3d} {e["key"][1] if e["key"] else -1:5d} '
			f'{e["pitch"]:6d}   #{r["src"]:<4d} p{s["pitch"]:<4d} '
			f'{str(s.get("key")):14s} {str(r["key"]):14s}')


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--root', default=DEFAULT_ROOT)
	ap.add_argument('--sample', default=None,
		help='basename or a unique prefix; default is the first pair by sorted name')
	ap.add_argument('--worst', type=int, default=0,
		help='instead draw the N worst pairs by @tick precision (scans every pair first)')
	ap.add_argument('--from-measure', type=int, default=None)
	ap.add_argument('--to-measure', type=int, default=None)
	ap.add_argument('--measures', type=int, default=None,
		help="draw only the first N bars of the window (the window's own first bar, which is not "
			'always bar 1); scoring still covers the whole window')
	ap.add_argument('--src-window', type=int, default=SRC_WINDOW)
	ap.add_argument('--out', default=DEFAULT_OUT)
	ap.add_argument('--verbose', action='store_true', help='list the wrong matches')
	args = ap.parse_args()

	src_dir = resolve_dir(args.root, SRC_ARM)
	tgt_dir = resolve_dir(args.root, TGT_ARM)
	if not src_dir or not tgt_dir:
		print(f'no corpus: need {SRC_ARM} and {TGT_ARM} under {args.root}')
		return 1

	load_notes.tokenizer = Midiseq2Tokenizer()
	load_notes.keywords = keyword_tokens(load_notes.tokenizer)

	names = sorted(set(os.listdir(src_dir)) & set(os.listdir(tgt_dir)))
	if not names:
		print('no shared basenames between the two arms')
		return 1

	if args.worst:
		# Scan every pair, then draw the weakest. This is the entry point for "show me where it fails"
		# without having to guess a filename -- the aggregate check says the corpus has a weak tail
		# (@tick precision p10 0.384) and this is how to look at it.
		scored = []
		for name in names:
			src, tgt = build_pair(src_dir, tgt_dir, name, args.src_window)
			if src is None:
				continue
			_state, records = align_and_judge(src, tgt)
			stats = summarise(records)
			if stats['precision'] is not None:
				scored.append((stats['precision'], name))
		scored.sort()
		chosen = [n for _p, n in scored[:args.worst]]
		print(f'{len(scored)} pairs scanned; worst {len(chosen)} by @tick precision:')
		for p, n in scored[:args.worst]:
			print(f'  {p:.4f}  {n}')
	elif args.sample:
		chosen = [n for n in names if n == args.sample or n.startswith(args.sample)]
		if not chosen:
			print(f'no pair matching {args.sample!r}; e.g. {names[0]}')
			return 1
		if len(chosen) > 1:
			print(f'{args.sample!r} matches {len(chosen)} pairs; using {chosen[0]}')
			chosen = chosen[:1]
	else:
		chosen = names[:1]

	drawn = 0
	for name in chosen:
		src, tgt = build_pair(src_dir, tgt_dir, name, args.src_window)
		if src is None:
			print(f'{name[:8]}: too short to measure')
			continue
		_state, records = align_and_judge(src, tgt)
		stats = summarise(records)
		print(f'{name}')
		report(src, tgt, records, stats, args.verbose)
		stem = name.split('.')[0]
		path = os.path.join(args.out, f'{stem}.align.png')
		title = f'{stem}  —  {SRC_ARM} -> {TGT_ARM}, src_window {args.src_window} tokens'
		if plot(src, tgt, records, stats, path, title, args.from_measure, args.to_measure,
				args.measures):
			print(f'  wrote {path}')
			drawn += 1
	return 0 if drawn else 1


if __name__ == '__main__':
	sys.exit(main())
