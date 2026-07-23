'''Onset comparison — generated midiseq2 text vs source lilylet score.

Compares note_on onsets from GENERATED midiseq2 text files against the corresponding
lilylet score's onsets (from the AST server). Unlike onset_compare_interactive.py (which
reads tensor-based dataset stores), this script works with the plain-text .midiseq2.txt
output of gen_midiseq2_prefixtl_from_lilylet.py.

For each measure it overlays (x = onset normalized within the bar; y = MIDI pitch):
  - lilylet onsets  (blue o) : from the lilylet AST server (POST /onsets) -> onsetNorm x pitch
  - MIDI note_on    (red x)  : from the generated midiseq2 text, parsed per-measure via
                                Midiseq2Tokenizer.parse_events, ticks normalized within the
                                measure's [0, last_tick) span.

Prereq: start the AST server first (from the lilylet repo) —
  cd ~/work/lilylet && npx tsx tools/astServer.ts --port 8788

Usage:
  python3 tests/midi/onset_compare_gen_midiseq2.py --gen-dir tests/output/midiseq2_prefixtl_from_lilylet \
      --lyl-dir ~/work/lilylet/tests/output/notagenx-from-abc
  python3 tests/midi/onset_compare_gen_midiseq2.py --gen-dir <dir> --lyl-dir <dir> --no-block
'''

import argparse
import collections
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = next(p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents] if (p / 'starry').is_dir())
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

import matplotlib
_INTERACTIVE = True
if matplotlib.get_backend().lower() == 'agg':
	for _bk in ('TkAgg', 'QtAgg', 'Qt5Agg'):
		try:
			matplotlib.use(_bk, force=True)
			break
		except Exception:		# noqa: BLE001
			continue
	else:
		_INTERACTIVE = False
import matplotlib.pyplot as plt
import numpy as np

from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer

AST_SERVER = 'http://127.0.0.1:8788'


# ---------------------------------------------------------------------------------------
# lilylet onsets (AST server) — {measure_index: [(onsetNorm, pitch), ...]}
# ---------------------------------------------------------------------------------------

def query_lyl_onsets (lyl_code: str):
	req = urllib.request.Request(AST_SERVER + '/onsets',
		data=json.dumps({'code': lyl_code}).encode(), headers={'Content-Type': 'application/json'})
	resp = json.load(urllib.request.urlopen(req, timeout=30))
	assert resp.get('ok'), resp.get('error')
	by_measure = {}
	for m in resp['measures']:
		by_measure[m['index']] = [(n['onsetNorm'], p) for n in m['notes'] for p in n['midi']]
	return by_measure


# ---------------------------------------------------------------------------------------
# MIDI note_on onsets from generated midiseq2 text — per-measure parse
# ---------------------------------------------------------------------------------------

def _seq2_body_pitch_vel (body):
	'''From a midiseq2 note_on body -> (pitch, vel).'''
	pitch, vel = 0, 0
	for tok in body[1:]:
		if tok.startswith('#'):
			try:
				pitch = int(tok[1:], 16)
			except ValueError:
				pitch = 0
		elif tok.startswith('$'):
			try:
				vel = int(tok[1:], 16)
			except ValueError:
				vel = 0
	return pitch, vel


def midi_onsets_from_gen_text (midiseq2_text: str, tk: Midiseq2Tokenizer):
	'''Parse generated midiseq2 text (one line per measure) into per-measure note_on onsets.

	Each line represents one measure (line 0 = header/measure 0, line 1+ = body measures).
	parse_events accumulates ticks measure-locally (since the generator re-sets per measure),
	so we parse each line independently and normalize within [0, max_tick] of that measure.

	Returns {measure_index: [(onsetNorm, pitch), ...]}
	'''
	lines = midiseq2_text.strip().split('\n')
	per_measure = {}
	for meas_idx, line in enumerate(lines):
		line = line.strip()
		if not line:
			continue
		_, events = tk.parse_events(line)
		if not events:
			continue
		# collect note_on with vel > 0
		notes = []
		for abst, is_off, body in events:
			if not is_off and body and body[0] == 'note_on':
				pitch, vel = _seq2_body_pitch_vel(body)
				if vel > 0:
					notes.append((abst, pitch))
		if not notes:
			continue
		# normalize ticks within this measure's span
		max_tick = max(abst for abst, _, _ in events) if events else 0
		if max_tick == 0:
			# all events at tick 0 — single onset
			per_measure[meas_idx] = [(0.0, p) for _, p in notes]
		else:
			per_measure[meas_idx] = [(t / max_tick, p) for t, p in notes]
	return per_measure


# ---------------------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------------------

def draw_measure (ax, meas, lyl_points, midi_points, name):
	ax.clear()
	if lyl_points:
		lx, ly = zip(*lyl_points)
		ax.scatter(lx, ly, s=90, marker='o', facecolors='none', edgecolors='tab:blue',
			linewidths=1.6, label=f'lilylet ({len(lyl_points)})')
	if midi_points:
		mx, my = zip(*midi_points)
		ax.scatter(mx, my, s=36, marker='x', color='tab:red',
			label=f'gen midi ({len(midi_points)})')
	ax.set_xlabel('onset (normalized within measure)')
	ax.set_ylabel('MIDI pitch')
	ax.set_title(f'{name}\nmeasure {meas}: lilylet (o) vs generated midi (x)')
	ax.set_xlim(-0.05, 1.05)
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best')
	# pitch-set overlap
	lyl_p = {p for _, p in lyl_points} if lyl_points else set()
	midi_p = {p for _, p in midi_points} if midi_points else set()
	ov = (len(lyl_p & midi_p) / len(lyl_p)) if lyl_p else float('nan')
	return ov


def draw_summary (fig, overlaps, name, out_path):
	'''Draw a summary bar chart of per-measure pitch-set overlap + stats.'''
	fig.clear()
	ax1 = fig.add_subplot(2, 1, 1)
	measures = sorted(overlaps.keys())
	vals = [overlaps[m] for m in measures]
	valid = [v for v in vals if not (v != v)]		# filter nan
	ax1.bar(range(len(measures)), vals, color='tab:green', alpha=0.7, width=0.8)
	ax1.axhline(np.mean(valid) if valid else 0, color='tab:orange', ls='--', lw=1.5,
		label=f'mean={np.mean(valid):.2f}' if valid else '')
	ax1.set_xlabel('measure index')
	ax1.set_ylabel('pitch-set overlap')
	ax1.set_title(f'{name} — per-measure pitch overlap (lilylet ∩ gen_midi / lilylet)')
	ax1.set_ylim(0, 1.05)
	ax1.legend(loc='lower right')
	ax1.grid(True, alpha=0.3)

	# stats text
	ax2 = fig.add_subplot(2, 1, 2)
	ax2.axis('off')
	stats = (
		f'Measures compared: {len(measures)}\n'
		f'Mean overlap: {np.mean(valid):.3f}\n'
		f'Median overlap: {np.median(valid):.3f}\n'
		f'Measures with overlap >= 0.5: {sum(1 for v in valid if v >= 0.5)} / {len(valid)}\n'
		f'Measures with overlap == 0: {sum(1 for v in valid if v == 0)} / {len(valid)}'
	)
	ax2.text(0.1, 0.5, stats, fontsize=12, family='monospace', va='center',
		transform=ax2.transAxes)
	fig.tight_layout()
	fig.savefig(out_path, bbox_inches='tight', dpi=130)
	print(f'  summary -> {out_path}')


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------

def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--gen-dir', required=True,
		help='directory of generated .midiseq2.txt files')
	ap.add_argument('--lyl-dir', required=True,
		help='directory of source .lyl files (matched by filename stem)')
	ap.add_argument('--out-dir', default=None,
		help='output directory for PNGs (default: <gen-dir>/onset_compare)')
	ap.add_argument('--max-measures', type=int, default=0,
		help='max measures to plot per song (0 = all)')
	ap.add_argument('--no-block', action='store_true',
		help='save PNGs non-interactively (default if no GUI backend)')
	args = ap.parse_args()

	gen_dir = Path(args.gen_dir)
	lyl_dir = Path(args.lyl_dir)
	out_dir = Path(args.out_dir) if args.out_dir else gen_dir / 'onset_compare'
	assert gen_dir.is_dir(), f'--gen-dir not found: {gen_dir}'
	assert lyl_dir.is_dir(), f'--lyl-dir not found: {lyl_dir}'

	gen_files = sorted(gen_dir.glob('*.midiseq2.txt'))
	assert gen_files, f'no .midiseq2.txt files in {gen_dir}'

	if not _INTERACTIVE:
		args.no_block = True
	out_dir.mkdir(parents=True, exist_ok=True)

	tk = Midiseq2Tokenizer()
	print(f'gen-dir : {gen_dir}  ({len(gen_files)} files)')
	print(f'lyl-dir : {lyl_dir}')
	print(f'out-dir : {out_dir}')

	for gi, gf in enumerate(gen_files):
		stem = gf.stem.replace('.midiseq2', '')
		# find matching .lyl source
		lyl_path = lyl_dir / (stem + '.lyl')
		if not lyl_path.exists():
			print(f'\n[{gi+1}/{len(gen_files)}] {stem} — SKIP (no matching .lyl)')
			continue
		print(f'\n===== [{gi+1}/{len(gen_files)}] {stem} =====')

		# parse generated midiseq2 text
		gen_text = gf.read_text(encoding='utf-8')
		midi_by_measure = midi_onsets_from_gen_text(gen_text, tk)

		# query lilylet AST server for reference onsets
		lyl_text = lyl_path.read_text(encoding='utf-8')
		try:
			lyl_by_measure = query_lyl_onsets(lyl_text)
		except Exception as e:		# noqa: BLE001
			print(f'  AST server error: {e}')
			print(f'  Start it first: cd ~/work/lilylet && npx tsx tools/astServer.ts --port 8788')
			return

		# measures present in both
		common = sorted(set(lyl_by_measure) & set(midi_by_measure))
		if not common:
			print(f'  no common measures (lyl has {sorted(lyl_by_measure.keys())[:5]}..., '
				f'midi has {sorted(midi_by_measure.keys())[:5]}...)')
			continue
		if args.max_measures:
			common = common[:args.max_measures]
		print(f'  lyl measures: {len(lyl_by_measure)} | gen midi measures: {len(midi_by_measure)} | common: {len(common)}')

		# per-measure plots + overlap tracking
		overlaps = {}
		song_dir = out_dir / stem
		song_dir.mkdir(parents=True, exist_ok=True)
		fig, ax = plt.subplots(figsize=(11, 5))
		for meas in common:
			lyl_pts = lyl_by_measure.get(meas, [])
			midi_pts = midi_by_measure.get(meas, [])
			ov = draw_measure(ax, meas, lyl_pts, midi_pts, stem[:40])
			overlaps[meas] = ov
			path = song_dir / f'm{meas:03d}.png'
			fig.savefig(path, bbox_inches='tight', dpi=110)

		# summary chart
		fig_sum = plt.figure(figsize=(12, 6))
		summary_path = song_dir / '_summary.png'
		draw_summary(fig_sum, overlaps, stem[:50], summary_path)

		valid_ov = [v for v in overlaps.values() if v == v]
		mean_ov = np.mean(valid_ov) if valid_ov else 0
		print(f'  saved {len(common)} measure PNGs + summary to {song_dir}/')
		print(f'  mean pitch-overlap: {mean_ov:.3f} ({len(valid_ov)} measures)')

		plt.close('all')

	print('\ndone.')


if __name__ == '__main__':
	main()
