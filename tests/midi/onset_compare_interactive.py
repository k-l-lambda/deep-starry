'''Interactive per-measure onset comparison — lilylet score vs dataset MIDI.

Config-driven sibling of tests/midi/midiMeasurewise_inspect.ipynb §6 and
tests/midi/lyl2midi_onset_compare.ipynb, but as a standalone matplotlib script that walks a
sample's measures ONE AT A TIME: it draws measure m's overlay, then waits for Enter to advance to
the next measure (q + Enter quits, a number jumps to that measure).

Reads a MIDI-measurewise dataset straight from a training config, dispatching on `data.type`:
  - CondMidiPatchy      (configs/midi-measurewise.yaml)          — basic MidiText event patches,
                                                                    decoded via MidiTokenizer.
  - Seq2CondMidiPatchy  (configs/midi-measurewise-seq2cond.yaml) — midiseq2 token patches, decoded
                                                                    via Midiseq2Tokenizer.

For each measure it overlays (x = onset normalized within the bar, 0=barline .. 1=next barline;
y = MIDI pitch):
  - lilylet onsets  (blue o) : from the lilylet AST server (POST /onsets) -> onsetNorm x pitch.
  - MIDI note_on    (red x)  : reconstructed by accumulating deltaTime / elapse over the sample's
                               midi patches, bucketed by the patch's OWN measure, normalized within
                               each played measure's [start, next-start) tick span. A lilylet measure
                               maps (via src_measure) to the played midi measure(s) — under expanded
                               repeats that can be more than one.

Prereq: start the AST server first (from the lilylet repo) —
  cd ~/work/lilylet && npx tsx tools/astServer.ts --port 8788

Usage:
  python3 tests/midi/onset_compare_interactive.py [--config CONFIG] [--data-dir DIR]
      [--sample N | --id SUBSTR] [--start-measure M] [--no-block]
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
# Pick an interactive backend so the per-measure figure can be shown and refreshed while we block
# on Enter. The repo default is the headless 'agg' (can't show a window); fall back to it (and to
# saving PNGs) only if no GUI backend is importable.
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

from starry.utils.config import Configuration
from starry.lilylet.data.patchifier import LilyletTokenizer

AST_SERVER = 'http://127.0.0.1:8788'


# ---------------------------------------------------------------------------------------
# config -> store (dispatch on data.type: CondMidiPatchy | Seq2CondMidiPatchy)
# ---------------------------------------------------------------------------------------

def open_store (config_path: str, data_dir: str):
	'''Resolve data.root against DATA_DIR and open the matching _ItemStore.

	Returns (store, kind) where kind is 'basic' (CondMidiPatchy, MidiText events) or 'seq2'
	(Seq2CondMidiPatchy, midiseq2 tokens) — the two share an item schema and differ only in how a
	midi patch decodes to a MidiText line. `ds_dir` (the artifact's directory) is where the source
	`lyl/<id>.lyl` files live for the AST server.
	'''
	config = Configuration.createOrLoad(config_path, volatile=True)
	dtype = config['data.type']
	root = os.path.join(data_dir, config['data.root'])
	assert os.path.exists(root), f'data.root not found: {root}'
	ds_dir = os.path.dirname(root)
	if dtype == 'CondMidiPatchy':
		from starry.midi.data.condPatchy import _get_store
		return _get_store(root), 'basic', ds_dir
	if dtype == 'Seq2CondMidiPatchy':
		from starry.midi.data.seq2CondPatchy import _get_store
		return _get_store(root), 'seq2', ds_dir
	raise SystemExit(f'unsupported data.type {dtype!r} (want CondMidiPatchy or Seq2CondMidiPatchy)')


def pick_sample (store, sample: int, id_substr: str):
	'''Select the item index: explicit --sample, or first id containing --id, else 0.'''
	if sample is not None:
		assert 0 <= sample < len(store), f'--sample {sample} out of range [0,{len(store)})'
		return sample
	if id_substr:
		for i in range(len(store)):
			if id_substr in str(store.get(i).get('id', '')):
				return i
		raise SystemExit(f'no sample id contains {id_substr!r}')
	return 0


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
# MIDI note_on onsets, reconstructed from the sample's midi patches -> {own_measure: [(tick, pitch)]}
# ---------------------------------------------------------------------------------------

def midi_onsets_basic (item, mt):
	'''CondMidiPatchy: one event per patch, decoded to a MidiText line via MidiTokenizer.

	Accumulate deltaTime over the midi segment (patches[L:]) to get absolute tick; bucket note_on
	(vel>0) by the patch's OWN measure. Skips <eom> boundary patches (they carry no deltaTime).
	Mirrors midiMeasurewise_inspect.ipynb §6.
	'''
	L = int(item['lyl_count'])
	T = item['patches'].shape[0]
	own = item['measures'].long()
	patches = item['patches'].long()
	abst = 0
	per_measure = collections.defaultdict(list)		# own measure -> [(abs_tick, pitch)]
	first_tick = {}									# own measure -> smallest abs_tick
	for i in range(L, T):
		if mt.eom_id is not None and int(patches[i, 0]) == mt.eom_id:
			continue
		line = mt.decode_event(patches[i].tolist())
		parts = line.split()
		if len(parts) < 2:
			continue
		try:
			delta = int(parts[1], 16)
		except ValueError:
			delta = 0
		abst += delta
		mi = int(own[i])
		first_tick.setdefault(mi, abst)
		if parts[0] == 'note_on' and len(parts) >= 5:
			try:
				vel = int(parts[4], 16)
			except ValueError:
				vel = 0
			if vel > 0:
				try:
					pitch = int(parts[3], 16)
				except ValueError:
					pitch = 0
				per_measure[mi].append((abst, pitch))
	return per_measure, first_tick


def _seq2_body_pitch_vel (body):
	'''From a midiseq2 note_on body (['note_on', 'C..'?, '#pitch', '$vel'?]) -> (pitch, vel).

	Channel C0 / zero arg3 / zero arg4 are omitted by the encoder, so read by prefix tag, not
	position: pitch from the `#..` token (arg3), velocity from the `$..` token (arg4, absent => 0).
	'''
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


def midi_onsets_seq2 (item, tk):
	'''Seq2CondMidiPatchy: each midi patch is a run of midiseq2 token ids (a measure, chunked).

	Recover the token strings per patch (id -> token), regroup into the measure's midiseq2 text,
	parse to absolute-tick events via Midiseq2Tokenizer.parse_events, and bucket note_on (vel>0) by
	the patch's OWN measure. parse_events already accumulates each event's leading E… elapse into an
	absolute tick FROM the measure start (the patchifier re-times each measure self-contained), so
	ticks are measure-local — exactly what we normalize within [0, bar-span).
	'''
	L = int(item['lyl_count'])
	T = item['patches'].shape[0]
	own = item['measures'].long()
	patches = item['patches'].long()
	drop = {tk.pad_id, tk.bos_id, tk.eos_id, tk.eom_id}
	# regroup patch rows by own measure, decode ids -> token strings (skipping specials).
	toks_by_measure = collections.defaultdict(list)
	for i in range(L, T):
		mi = int(own[i])
		for tid in patches[i].tolist():
			if tid in drop:
				continue
			toks_by_measure[mi].append(tk.tokens[tid] if 0 <= tid < len(tk.tokens) else '<unknown>')
	per_measure = collections.defaultdict(list)		# own measure -> [(abs_tick, pitch)]
	first_tick = {}
	for mi, toks in toks_by_measure.items():
		_, events = tk.parse_events(' '.join(toks))
		for abst, is_off, body in events:
			first_tick.setdefault(mi, abst)
			first_tick[mi] = min(first_tick[mi], abst)
			if body and body[0] == 'note_on':
				pitch, vel = _seq2_body_pitch_vel(body)
				if vel > 0:
					per_measure[mi].append((abst, pitch))
	return per_measure, first_tick


def normalize_onsets (per_measure, first_tick, played_measures):
	'''Normalize note_on ticks within each played measure's [start, next-start) span -> points.

	Returns [(onsetNorm, pitch)] pooled over `played_measures` (the midi measures a lilylet measure
	maps to via src_measure — >1 under expanded repeats).
	'''
	sorted_m = sorted(first_tick)
	def span (mi):
		start = first_tick[mi]
		later = [first_tick[m] for m in sorted_m if first_tick[m] > start]
		end = min(later) if later else max((t for pts in per_measure.values() for t, _ in pts), default=start + 1)
		return start, max(end, start + 1)
	points = []
	for mi in played_measures:
		if mi not in first_tick:
			continue
		start, end = span(mi)
		d = end - start
		for t, pitch in per_measure.get(mi, []):
			points.append(((t - start) / d, pitch))
	return points


# ---------------------------------------------------------------------------------------
# interactive per-measure walk
# ---------------------------------------------------------------------------------------

def draw_measure (ax, meas, lyl_points, midi_points, played, name):
	ax.clear()
	if lyl_points:
		lx, ly = zip(*lyl_points)
		ax.scatter(lx, ly, s=90, marker='o', facecolors='none', edgecolors='tab:blue',
			linewidths=1.6, label=f'lilylet ({len(lyl_points)})')
	if midi_points:
		mx, my = zip(*midi_points)
		ax.scatter(mx, my, s=36, marker='x', color='tab:red',
			label=f'midi note_on ({len(midi_points)})')
	repeat = '  (repeated: played %s)' % played if len(played) > 1 else ''
	ax.set_xlabel('onset (normalized within measure, 0=barline .. 1=next barline)')
	ax.set_ylabel('MIDI pitch number')
	ax.set_title(f'{name}\nmeasure {meas}: lilylet (o) vs MIDI (x){repeat}')
	ax.set_xlim(-0.05, 1.05)
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best')
	# per-measure pitch-set overlap, printed alongside the plot.
	lyl_p = {p for _, p in lyl_points}
	midi_p = {p for _, p in midi_points}
	ov = (len(lyl_p & midi_p) / len(lyl_p)) if lyl_p else float('nan')
	return ov


def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--config', default=str(REPO_ROOT / 'configs' / 'midi-measurewise.yaml'),
		help='training config (CondMidiPatchy or Seq2CondMidiPatchy); default midi-measurewise.yaml')
	ap.add_argument('--data-dir', default=str(Path.home() / 'data'),
		help='DATA_DIR base that data.root is relative to (default ~/data)')
	ap.add_argument('--sample', type=int, default=None, help='item index to inspect (default 0)')
	ap.add_argument('--id', default='', help='pick the first sample whose id contains this substring')
	ap.add_argument('--start-measure', type=int, default=None, help='first measure to show')
	ap.add_argument('--no-block', action='store_true',
		help='save each measure to tests/output/onset_compare/ as PNG instead of waiting for Enter')
	args = ap.parse_args()

	store, kind, ds_dir = open_store(args.config, args.data_dir)
	ex = pick_sample(store, args.sample, args.id)
	item = store.get(ex)
	name = str(item['id'])
	L = int(item['lyl_count'])
	T = item['patches'].shape[0]
	own = item['measures'].long()
	src = item['src_measures'].long()
	print(f'config : {args.config}')
	print(f'kind   : {kind}   store items: {len(store)}')
	print(f'sample #{ex}: id={name}')
	print(f'  T={T}  lyl_count L={L}  midi patches={T - L}')

	# lilylet source + AST onsets (per measure).
	lyl_src = os.path.join(ds_dir, 'lyl', name + '.lyl')
	assert os.path.exists(lyl_src), f'lyl source not found: {lyl_src}'
	try:
		lyl_by_measure = query_lyl_onsets(open(lyl_src, encoding='utf-8').read())
	except Exception as e:		# noqa: BLE001
		raise SystemExit(f'AST server query failed ({e}). Start it first:\n'
			f'  cd ~/work/lilylet && npx tsx tools/astServer.ts --port 8788')

	# MIDI note_on onsets (by own measure), decoder chosen by store kind.
	if kind == 'basic':
		from starry.midi.tokenizer import MidiTokenizer
		per_measure, first_tick = midi_onsets_basic(item, MidiTokenizer())
	else:
		from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer
		per_measure, first_tick = midi_onsets_seq2(item, Midiseq2Tokenizer())

	# lilylet body measures present in this sample (walk these in order).
	lyl_measures = sorted(set(int(m) for m in own[:L].tolist() if m > 0))
	# map each lilylet measure -> played midi measure(s) via src on the midi segment.
	played_of = collections.defaultdict(set)
	for i in range(L, T):
		played_of[int(src[i])].add(int(own[i]))
	walk = [m for m in lyl_measures if m in lyl_by_measure and lyl_by_measure[m]]
	if args.start_measure is not None:
		walk = [m for m in walk if m >= args.start_measure]
	if not walk:
		raise SystemExit('no measures with lilylet onsets to show')
	print(f'  measures to walk: {walk[0]}..{walk[-1]} ({len(walk)} measures)')

	if args.no_block:
		out_dir = REPO_ROOT / 'tests' / 'output' / 'onset_compare'
		out_dir.mkdir(parents=True, exist_ok=True)
	if not _INTERACTIVE and not args.no_block:
		print('WARNING: no interactive matplotlib backend available; falling back to --no-block (saving PNGs).')
		args.no_block = True
		out_dir = REPO_ROOT / 'tests' / 'output' / 'onset_compare'
		out_dir.mkdir(parents=True, exist_ok=True)

	if not args.no_block:
		plt.ion()
	fig, ax = plt.subplots(figsize=(11, 5))

	idx = 0
	while 0 <= idx < len(walk):
		meas = walk[idx]
		played = sorted(played_of.get(meas, {meas}))
		midi_points = normalize_onsets(per_measure, first_tick, played)
		ov = draw_measure(ax, meas, lyl_by_measure[meas], midi_points, played, name)
		print(f'[{idx + 1}/{len(walk)}] measure {meas:>3}: '
			f'lyl={len(lyl_by_measure[meas])} midi={len(midi_points)} '
			f'pitch-overlap={ov:.2f}  played={played}')
		if args.no_block:
			path = out_dir / f'{ex:04d}_m{meas:03d}.png'
			fig.savefig(path, bbox_inches='tight', dpi=110)
			print('    saved', path)
			idx += 1
			continue
		fig.canvas.draw(); fig.canvas.flush_events()
		cmd = input('  Enter=next | p=prev | <number>=jump | q=quit > ').strip().lower()
		if cmd in ('q', 'quit'):
			break
		if cmd in ('p', 'prev', 'b', 'back'):
			idx = max(0, idx - 1)
		elif cmd.isdigit():
			want = int(cmd)
			idx = next((k for k, m in enumerate(walk) if m >= want), idx)
		else:
			idx += 1

	if not args.no_block:
		plt.ioff()
	print('done.')


if __name__ == '__main__':
	main()
