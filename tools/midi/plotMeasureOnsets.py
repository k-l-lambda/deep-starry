'''Sample N random measures from a CondMidiPatchy dataset and save lilylet-vs-MIDI onset plots.

Mirrors tests/midi/midiMeasurewise_inspect.ipynb §6, in batch: for each sampled (item, lyl
measure), query the lilylet AST server (tools/astServer.ts, /onsets) for the lilylet note
onsets, reconstruct the MIDI note_on onsets from the packed event patches' deltaTimes, and
save a scatter (x = onset normalized within the measure, y = MIDI pitch).

Usage (run from the deep-starry repo root, with PYTHONPATH=.):
  python3 tools/midi/plotMeasureOnsets.py \
    --root /data1/datasets/nota/midi-measurewise/nota20260701/cond-midi.lmmw.pt \
    --lyl-root /data1/datasets/nota/lilylet/lyl \
    --out-dir /data1/datasets/nota/midi-measurewise/nota20260701/onset-tests \
    --n 100 --server http://127.0.0.1:8788 --seed 20260701
'''

import argparse
import json
import os
import random
import sys
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.midi.data.condPatchy import _get_store
from starry.midi.tokenizer import MidiTokenizer, FIELD_EVENT_TOKENS

# FIELD events carry deltaTime as their first value; HEADER events (ticks_per_beat / format_type
# / track) carry a single value and NO deltaTime, so they must be excluded from tick accumulation.
_FIELD_EVENTS = set(FIELD_EVENT_TOKENS)


def build_lyl_index (lyl_root):
	'''Map bare stem -> absolute .lyl path by walking the (possibly nested) lyl tree.'''
	index = {}
	for dp, _, files in os.walk(lyl_root):
		for f in files:
			if f.endswith('.lyl'):
				index[f[:-4]] = os.path.join(dp, f)
	return index


def query_onsets (server, code):
	req = urllib.request.Request(server + '/onsets',
		data=json.dumps({'code': code}).encode(), headers={'Content-Type': 'application/json'})
	return json.load(urllib.request.urlopen(req, timeout=60))


def midi_onsets_by_measure (item, mt):
	'''Reconstruct per-(own)measure MIDI note_on onsets from the packed midi patches.

	Returns (per_measure_onsets, measure_first_tick):
	  per_measure_onsets[own_measure] -> list of (abs_tick, midi_pitch)
	  measure_first_tick[own_measure]  -> first abs_tick seen in that measure
	'''
	patches = item['patches'].long()
	modality = item['modality']
	own = item['measures'].long()
	L = int(item['lyl_count'])
	T = patches.shape[0]

	abst = 0
	per_measure = {}
	first_tick = {}
	for i in range(L, T):
		line = mt.decode_event(patches[i].tolist())
		if not line or int(patches[i, 0]) == mt.eom_id:
			continue
		parts = line.split()
		if len(parts) < 2:
			continue
		# Header events (ticks_per_beat / format_type / track) carry a SINGLE value and NO
		# deltaTime — their parts[1] is that value, not a time gap. Adding it corrupts the
		# running tick (e.g. "ticks_per_beat 1e0" would inject +480). Only FIELD events carry
		# deltaTime as parts[1], so skip anything not in the field-event set for accumulation.
		if parts[0] not in _FIELD_EVENTS:
			continue
		try:
			abst += int(parts[1], 16)
		except ValueError:
			pass
		mi = int(own[i])
		first_tick.setdefault(mi, abst)
		if parts[0] == 'note_on' and len(parts) >= 5:
			try:
				vel = int(parts[4], 16)
			except ValueError:
				vel = 0
			if vel > 0:
				per_measure.setdefault(mi, []).append((abst, int(parts[3], 16)))
	return per_measure, first_tick


def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--root', required=True, help='cond-midi.lmmw.pt index path')
	ap.add_argument('--lyl-root', required=True, help='root of source .lyl files (walked recursively)')
	ap.add_argument('--out-dir', required=True, help='directory to write onset PNGs into')
	ap.add_argument('--n', type=int, default=100, help='number of measures to sample')
	ap.add_argument('--server', default='http://127.0.0.1:8788', help='lilylet AST server base url')
	ap.add_argument('--seed', type=int, default=20260701)
	ap.add_argument('--dataset', default=None,
		help='dataset.yaml with authoritative per-measure start_tick (recommended; the '
		     'reconstructed first-event tick can miss the true bar line by a note). If omitted, '
		     'measure spans are derived from event ticks.')
	args = ap.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	# authoritative measure start_tick per sample id (from dataset.yaml), if provided.
	start_ticks = {}
	if args.dataset:
		import yaml
		try:
			Loader = yaml.CSafeLoader
		except AttributeError:
			Loader = yaml.SafeLoader
		print('loading measure start_ticks from', args.dataset, '(this may take a minute)...')
		d = yaml.load(open(args.dataset, encoding='utf-8'), Loader=Loader)
		for s in d['samples']:
			start_ticks[s['id']] = {int(m['index']): int(m['start_tick']) for m in s.get('measures', [])}
		print('start_tick maps for', len(start_ticks), 'samples')
	store = _get_store(args.root)
	mt = MidiTokenizer()
	rng = random.Random(args.seed)
	print('indexing lyl sources under', args.lyl_root, '...')
	lyl_index = build_lyl_index(args.lyl_root)
	print('lyl files indexed:', len(lyl_index), '| store items:', len(store))

	# cache: item index -> (item, lyl onsets response, midi onsets)
	onset_cache = {}
	saved = 0
	attempts = 0
	summary = []
	max_attempts = args.n * 8

	while saved < args.n and attempts < max_attempts:
		attempts += 1
		ei = rng.randrange(len(store))
		if ei not in onset_cache:
			item = store.get(ei)
			lyl_path = lyl_index.get(item['id'])
			if not lyl_path:
				onset_cache[ei] = None
				continue
			with open(lyl_path, encoding='utf-8') as f:
				code = f.read()
			try:
				resp = query_onsets(args.server, code)
			except Exception as e:  # noqa: BLE001
				onset_cache[ei] = None
				continue
			if not resp.get('ok'):
				onset_cache[ei] = None
				continue
			pm, ft = midi_onsets_by_measure(item, mt)
			onset_cache[ei] = (item, {m['index']: m for m in resp['measures']}, pm, ft)
		cached = onset_cache[ei]
		if cached is None:
			continue
		item, lyl_measures, per_measure, first_tick = cached

		own = item['measures'].long()
		src = item['src_measures'].long()
		modality = item['modality']
		L = int(item['lyl_count'])
		T = item['patches'].shape[0]

		# lilylet body measures that also exist in the AST onsets
		lyl_body = sorted(set(int(m) for m in own[:L].tolist() if m > 0) & set(lyl_measures))
		if not lyl_body:
			continue
		MEAS = rng.choice(lyl_body)
		mrec = lyl_measures[MEAS]
		lyl_points = [(n['onsetNorm'], p) for n in mrec['notes'] for p in n['midi']]

		# midi played measures aligned to this lyl measure (src_measure == MEAS)
		midi_own = sorted(set(int(own[i]) for i in range(L, T) if int(src[i]) == MEAS))
		# Prefer the authoritative per-measure start_tick from dataset.yaml: the true bar line,
		# which the reconstructed first-event tick can miss by a note (a bar opening with a rest,
		# or a boundary note_off attributed to the previous measure). Fall back to first_tick.
		st_map = start_ticks.get(item['id'])
		def span (mi):
			if st_map and mi in st_map:
				start = st_map[mi]
				nxts = [st_map[m] for m in st_map if st_map[m] > start]
				end = min(nxts) if nxts else start + max((t for pts in per_measure.values() for t, _ in pts), default=start + 1)
				return start, max(end, start + 1)
			start = first_tick[mi]
			nxts = [first_tick[m] for m in first_tick if first_tick[m] > start]
			end = min(nxts) if nxts else max((t for pts in per_measure.values() for t, _ in pts), default=start + 1)
			return start, max(end, start + 1)
		midi_points = []
		for mi in midi_own:
			s0, e0 = span(mi)
			for t, pitch in per_measure.get(mi, []):
				midi_points.append(((t - s0) / (e0 - s0), pitch))

		if not lyl_points and not midi_points:
			continue

		fig, ax = plt.subplots(figsize=(10, 5))
		if lyl_points:
			lx, ly = zip(*lyl_points)
			ax.scatter(lx, ly, s=90, marker='o', facecolors='none', edgecolors='tab:blue',
				linewidths=1.6, label='lilylet (%d)' % len(lyl_points))
		if midi_points:
			mx, my = zip(*midi_points)
			ax.scatter(mx, my, s=36, marker='x', color='tab:red', label='midi note_on (%d)' % len(midi_points))
		ax.set_xlabel('onset (normalized within measure)')
		ax.set_ylabel('MIDI pitch number')
		ax.set_title('%s  m%d  (played midi %s)' % (item['id'][:20], MEAS, midi_own))
		ax.set_xlim(-0.05, 1.05)
		ax.grid(True, alpha=0.3)
		ax.legend(loc='best')
		fig.tight_layout()
		fname = '%04d_%s_m%03d.png' % (saved, item['id'][:16], MEAS)
		fig.savefig(os.path.join(args.out_dir, fname), dpi=90)
		plt.close(fig)
		summary.append(dict(file=fname, id=item['id'], measure=MEAS, played_midi=midi_own,
			lyl_points=len(lyl_points), midi_points=len(midi_points), timeSig=mrec['timeSig']))
		saved += 1
		if saved % 20 == 0:
			print('  saved %d/%d (attempts %d)' % (saved, args.n, attempts))

	with open(os.path.join(args.out_dir, 'index.json'), 'w') as f:
		json.dump(summary, f, indent=1)
	print('DONE: saved %d plots (%d attempts) -> %s' % (saved, attempts, args.out_dir))


if __name__ == '__main__':
	main()
