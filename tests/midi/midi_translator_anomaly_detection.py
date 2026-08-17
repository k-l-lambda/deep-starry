#!/usr/bin/env python3
'''Exhaustively map MidiTranslator loss/err back to fixed source regions.

The ordinary Seq2Seq2 validation split samples ONE crop per song. This diagnostic instead tiles every
source file from line 0 to EOF: each window is nominally 256 source lines, with its right edge extended
to the first following @tick boundary. The paired target range, wrappers, position ids and supervision
mask still come from Seq2Seq2, so every score has exactly the geometry the model was trained against.

Every window records its source and aligned-target line ranges plus teacher-forced next-token CE and
error rate. Raw CE sums and error/token counts are retained: corpus and song totals are token-weighted,
not a mean of window means. The JSON is the traceable result; the PNG only visualizes the per-window
distributions.

Default full run:
    python3 tests/midi/midi_translator_anomaly_detection.py

Short smoke run (explicitly reported as non-exhaustive):
    python3 tests/midi/midi_translator_anomaly_detection.py --device cpu --max-files 1 --max-windows 2
'''

import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import traceback

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt						# noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from starry.midi.data.seq2seq2 import (Seq2Seq2, _File, _get_file, _measures_in)  # noqa: E402
from starry.utils.config import Configuration								# noqa: E402
from starry.utils.model_factory import loadModel							# noqa: E402


DEFAULT_RUN = '/home/camus/data/models/deep-starry-logs/midi/20260814-midi-translator-nota1m-maiyi-l16d256'
DEFAULT_CORPUS = '/home/camus/data/midi/test202608/nota1m-100'
DEFAULT_OUT = os.path.join(REPO_ROOT, 'tests', 'output', 'midi_translator_anomaly_detection')


# --- pure helpers --------------------------------------------------------------------------

def percentile (values, q):
	'''Linear percentile without making NumPy a reporting dependency.'''
	if not values:
		return None
	ordered = sorted(float(v) for v in values)
	if len(ordered) == 1:
		return ordered[0]
	position = (len(ordered) - 1) * q
	lo = int(math.floor(position))
	hi = int(math.ceil(position))
	if lo == hi:
		return ordered[lo]
	return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def distribution (values):
	values = [float(v) for v in values]
	if not values:
		return dict(count=0, min=None, mean=None, median=None, p90=None, p95=None, p99=None, max=None)
	return dict(count=len(values), min=min(values), mean=sum(values) / len(values),
		median=statistics.median(values), p90=percentile(values, 0.90),
		p95=percentile(values, 0.95), p99=percentile(values, 0.99), max=max(values))


def aggregate_metrics (rows):
	'''Token-weighted CE and err from raw additive quantities.'''
	n_target = sum(int(row['n_target']) for row in rows)
	loss_sum = sum(float(row['loss_sum']) for row in rows)
	n_error = sum(int(row['n_error']) for row in rows)
	return dict(windows=len(rows), n_target=n_target, loss_sum=loss_sum, n_error=n_error,
		loss=loss_sum / n_target if n_target else None,
		err=n_error / n_target if n_target else None)


def tick_json (key):
	if key is None:
		return None
	if isinstance(key, tuple) and len(key) == 2:
		return {'measure': key[0], 'tick': key[1]}
	return {'measure': key}


def measure_json (rows):
	return [{'line': int(line), 'measure': int(measure)} for line, measure in rows]


def file_ticks (file, start, end):
	marks = [(line, key) for line, key in file.marks if start <= line < end]
	return dict(count=len(marks),
		first=({'line': marks[0][0], 'key': tick_json(marks[0][1])} if marks else None),
		last=({'line': marks[-1][0], 'key': tick_json(marks[-1][1])} if marks else None))


def iter_source_windows (source, nominal_lines=256):
	'''Yield a gap-free source tiling as dictionaries carrying Seq2Seq2 mark indices.

	The right edge is the first @tick mark at or AFTER start+nominal_lines. That mark is exclusive
	from the current range and becomes the next range's first mark, exactly matching `_bounds(a, z)`.
	'''
	if nominal_lines < 1:
		raise ValueError('nominal_lines must be positive')
	n_lines = len(source.lines)
	if n_lines == 0:
		return
	if not source.marks:
		yield dict(a=0, z=0, start=0, nominal_end=min(nominal_lines, n_lines), end=n_lines)
		return

	a = 0
	while True:
		start = 0 if a == 0 else source.marks[a][0]
		if start >= n_lines:
			break
		nominal_end = min(start + nominal_lines, n_lines)
		if nominal_end >= n_lines:
			z = len(source.marks)
			end = n_lines
		else:
			z = a
			while z < len(source.marks) and source.marks[z][0] < nominal_end:
				z += 1
			# `_bounds(a, z)` treats mark z as an exclusive boundary, and a == 0 is the special
			# head sentinel. If the first tick itself lies at/after the nominal boundary, using z=0
			# would make the next crop's a=0 point back at line 0 and stall. Consume that first mark
			# as part of the initial head range and use the following mark (or EOF) as the boundary.
			if z == a and a == 0:
				z = 1
			end = n_lines if z >= len(source.marks) else source.marks[z][0]
		yield dict(a=a, z=z, start=start, nominal_end=nominal_end, end=end)
		if z >= len(source.marks):
			break
		if z <= a or end <= start:
			raise AssertionError(f'window iterator stalled at mark {a} / line {start}')
		a = z


def validate_tiling (source, windows):
	if not source.lines:
		if windows:
			raise AssertionError('empty source unexpectedly produced a window')
		return
	if not windows or windows[0]['start'] != 0 or windows[-1]['end'] != len(source.lines):
		raise AssertionError('windows do not span source line 0 through EOF')
	for i, window in enumerate(windows):
		if window['end'] <= window['start']:
			raise AssertionError(f'empty window {i}')
		if i and windows[i - 1]['end'] != window['start']:
			raise AssertionError(f'gap/overlap between source windows {i - 1} and {i}')
		if window['end'] < window['nominal_end']:
			raise AssertionError(f'window {i} ended before its nominal boundary')
		if window['end'] < len(source.lines) and not source.lines[window['end']].startswith('@tick'):
			raise AssertionError(f'window {i} does not end on @tick')


def score_logits (pred, labels):
	'''Raw additive and per-token metrics for already-shifted supervised logits.'''
	n_target = int(labels.numel())
	if not n_target:
		raise ValueError('window has no supervised target tokens')
	loss_sum = float(F.cross_entropy(pred.float(), labels, reduction='sum').item())
	n_error = int((pred.argmax(dim=-1) != labels).sum().item())
	loss = loss_sum / n_target
	err = n_error / n_target
	if not math.isfinite(loss_sum) or not math.isfinite(loss) or not math.isfinite(err):
		raise ValueError('window produced non-finite loss/err')
	return dict(loss_sum=loss_sum, loss=loss, n_target=n_target, n_error=n_error, err=err)


def sha256_file (path):
	digest = hashlib.sha256()
	with open(path, 'rb') as f:
		while True:
			chunk = f.read(1024 * 1024)
			if not chunk:
				break
			digest.update(chunk)
	return digest.hexdigest()


def atomic_json (path, payload):
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	tmp = None
	try:
		with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=os.path.dirname(os.path.abspath(path)),
			prefix='.%s.' % os.path.basename(path), suffix='.tmp', delete=False) as f:
			tmp = f.name
			json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
			f.write('\n')
			f.flush()
			os.fsync(f.fileno())
		os.replace(tmp, path)
	except Exception:
		if tmp and os.path.exists(tmp):
			os.unlink(tmp)
		raise


# --- setup ---------------------------------------------------------------------------------

def validate_run (config, args):
	data = config['data.args'] or {}
	model = config['model.args'] or {}
	expected = {
		'model.type': (config['model.type'], 'MidiTranslator'),
		'data.type': (config['data.type'], 'Seq2Seq2'),
		'mark_mode': (data.get('mark_mode'), 'tick'),
		'pos_style': (data.get('pos_style'), 'sep'),
		'source_eom': (bool(data.get('source_eom')), False),
		'source_dir': (data.get('source_dir'), args.source_dir),
		'target_dir': (data.get('target_dir'), args.target_dir),
	}
	bad = ['%s=%r (expected %r)' % (key, got, want) for key, (got, want) in expected.items() if got != want]
	if bad:
		raise ValueError('run is incompatible with this scan: ' + '; '.join(bad))
	vocab_path = model.get('vocab_path')
	if not vocab_path or not os.path.isfile(vocab_path):
		raise ValueError(f'run-local vocab does not exist: {vocab_path!r}')
	if int(model.get('vocab_size') or 0) <= 0:
		raise ValueError('model.args.vocab_size is missing or invalid')
	if int(model.get('max_seq_len') or 0) <= 0:
		raise ValueError('model.args.max_seq_len is missing or invalid')
	return data, model, vocab_path


def load_run (args):
	config = Configuration.createOrLoad(args.run, volatile=True)
	data_args, model_args, vocab_path = validate_run(config, args)
	checkpoint_path = args.checkpoint or os.path.join(args.run, 'best.chkpt')
	if not os.path.isfile(checkpoint_path):
		raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
	checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
	if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model'), dict):
		raise ValueError('checkpoint must contain a model state dictionary')

	model = loadModel(config['model'], postfix='Loss', imports=config['imports'])
	model.deducer.load_state_dict(checkpoint['model'], strict=True)
	if model.weighted_loss:
		raise ValueError('this diagnostic defines loss as plain CE, but the run uses loss_type_weights')
	if model.deducer.vocab_size != int(model_args['vocab_size']):
		raise ValueError('constructed model vocabulary does not match the pinned config')
	model = model.to(args.device).eval()

	dataset = Seq2Seq2(args.corpus, '0/1', device=args.device, shuffle=False,
		source_dir=args.source_dir, target_dir=args.target_dir, mark_mode='tick',
		line_range=args.window_lines, p_head=0, p_tail=0, source_eom=False,
		max_tokens=0, resample_tries=1, align_retries=0, start_jitter=0,
		random_crop=False, seed=0, vocab_path=vocab_path, pos_style='sep', packed=False)
	if dataset.tokenizer.vocab_size != model.deducer.vocab_size:
		raise ValueError('feeder vocabulary does not match model embedding rows')

	source_names = set(dataset.source.names(args.source_dir))
	target_names = set(dataset.source.names(args.target_dir))
	shared = source_names & target_names
	if args.expect_files and (len(source_names) != args.expect_files or len(target_names) != args.expect_files
		or len(shared) != args.expect_files or source_names != target_names):
		raise ValueError('corpus contract failed: source=%d target=%d shared=%d source_only=%d target_only=%d '
			'(expected %d exact pairs)' % (len(source_names), len(target_names), len(shared),
				len(source_names - target_names), len(target_names - source_names), args.expect_files))

	identity = dict(directory=os.path.abspath(args.run), checkpoint=os.path.abspath(checkpoint_path),
		checkpoint_sha256=sha256_file(checkpoint_path), checkpoint_epoch=checkpoint.get('epoch'),
		vocab_path=os.path.abspath(vocab_path), vocab_sha256=sha256_file(vocab_path),
		model_type=config['model.type'], vocab_size=model.deducer.vocab_size,
		d_model=model_args.get('d_model'), n_layer=model_args.get('n_layer'),
		n_head=model_args.get('n_head'), max_seq_len=model_args.get('max_seq_len'),
		training_max_tokens=data_args.get('max_tokens'), pos_style=data_args.get('pos_style'))
	corpus = dict(root=os.path.abspath(args.corpus), source_arm=args.source_dir,
		target_arm=args.target_dir, source_files=len(source_names), target_files=len(target_names),
		shared_files=len(shared), source_only=sorted(source_names - target_names),
		target_only=sorted(target_names - source_names))
	return config, model, dataset, identity, corpus


# --- window metadata and evaluation --------------------------------------------------------

def alignment_metadata (dataset, source, target, a, z, align):
	if a <= 0:
		left = dict(requested=None, used=None, distance=0, clamped='start')
	else:
		used = dataset._walk_out(source, target, a, -1)
		left = dict(requested={'index': a, 'key': tick_json(source.marks[a][1])},
			used=({'index': used, 'key': tick_json(source.marks[used][1])} if used >= 0 else None),
			distance=(a - used if used >= 0 else a + 1), clamped=('start' if used < 0 else None))
	if z >= len(source.marks):
		right = dict(requested=None, used=None, distance=0, clamped='end')
	else:
		used = dataset._walk_out(source, target, z, 1)
		right = dict(requested={'index': z, 'key': tick_json(source.marks[z][1])},
			used=({'index': used, 'key': tick_json(source.marks[used][1])}
				if used < len(source.marks) else None),
			distance=(used - z if used < len(source.marks) else len(source.marks) - z),
			clamped=('end' if used >= len(source.marks) else None))
	return dict(range=list(align), left=left, right=right)


def window_record (dataset, name, song_index, window_index, source, target, window, align,
	ids, sep, positions, previous_target_end, train_cap, model_cap):
	a, z = window['a'], window['z']
	s_start, s_end = window['start'], window['end']
	t_start, t_end = align
	boundary = None if z >= len(source.marks) else {
		'line': source.marks[z][0], 'key': tick_json(source.marks[z][1])}
	target_overlap = max(0, previous_target_end - t_start) if previous_target_end is not None else 0
	target_gap = max(0, t_start - previous_target_end) if previous_target_end is not None else 0
	return {
		'id': '%s:w%04d' % (name.rsplit('.', 2)[0], window_index),
		'status': 'pending', 'song': name, 'song_index': song_index, 'window_index': window_index,
		'source': {
			'path': os.path.join(dataset.arm_source, name), 'total_lines': len(source.lines),
			'range': [s_start, s_end], 'lines': s_end - s_start,
			'nominal_end': window['nominal_end'], 'extension_lines': s_end - window['nominal_end'],
			'mark_range': [a, z], 'total_marks': len(source.marks), 'head': a <= 0,
			'tail': z >= len(source.marks), 'boundary': boundary,
			'ticks': file_ticks(source, s_start, s_end),
			'measures': measure_json(_measures_in(source, s_start, s_end)),
		},
		'target': {
			'path': os.path.join(dataset.arm_target, name), 'total_lines': len(target.lines),
			'range': [t_start, t_end], 'lines': t_end - t_start,
			'ticks': file_ticks(target, t_start, t_end),
			'measures': measure_json(_measures_in(target, t_start, t_end)),
			'overlap_with_previous': target_overlap, 'gap_from_previous': target_gap,
			'alignment': alignment_metadata(dataset, source, target, a, z, align),
		},
		'sequence': {
			'total_tokens': len(ids), 'source_tokens': sep, 'target_tokens': len(ids) - sep - 1,
			'sep_index': sep, 'unknown_tokens': ids.count(dataset.tokenizer.unknown_id),
			'position_min': min(positions), 'position_max': max(positions),
			'above_training_max_tokens': bool(train_cap and len(ids) > train_cap),
			'above_model_max_seq_len': bool(model_cap and len(ids) > model_cap),
		},
	}


def evaluate_window (dataset, model, ids, sep, positions):
	sample = (torch.tensor(ids, dtype=torch.long), sep,
		torch.tensor(positions, dtype=torch.long), 0)
	batch = dataset.collateBatch([sample])
	# These are the feeder/loss contracts this diagnostic relies on. Keep them live rather than merely
	# restating them in prose, because a future feeder edit must not silently change what the report means.
	if int(batch['target_mask'][0, :sep + 1].sum()) != 0:
		raise AssertionError('source or <sep> is supervised')
	if int(batch['target_mask'].sum()) != len(ids) - sep - 1:
		raise AssertionError('target mask does not cover the whole unjittered target')
	if ids[-1] != dataset.tokenizer.eos_id or not bool(batch['target_mask'][0, -1]):
		raise AssertionError('<eos> is not the final supervised token')
	logits = model.deducer(batch['input_ids'], batch['masks'], batch['position_ids'])
	pred, labels = model._shift(batch, logits)
	return score_logits(pred, labels), pred, labels


def error_json (kind, error, name=None, window_index=None):
	return dict(kind=kind, song=name, window_index=window_index,
		message='%s: %s' % (type(error).__name__, error), traceback=traceback.format_exc())


# --- report and plot -----------------------------------------------------------------------

def plot_report (path, windows, summaries):
	rows = [row for row in windows if row['status'] == 'ok']
	loss = [row['metrics']['loss'] for row in rows]
	err = [row['metrics']['err'] for row in rows]
	if not rows:
		return False
	fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
	for ax, values, title, color, xlabel, stat in (
		(axes[0], loss, 'Per-window cross-entropy', '#4C78A8', 'loss', summaries['loss']),
		(axes[1], err, 'Per-window token error rate', '#E45756', 'err', summaries['err'])):
		bins = min(60, max(10, round(math.sqrt(len(values)))))
		if min(values) == max(values):
			bins = 1
		ax.hist(values, bins=bins, color=color, edgecolor='white', linewidth=0.8)
		ax.axvline(stat['median'], color='#222222', linestyle='--', linewidth=1.2,
			label='median %.4g' % stat['median'])
		ax.axvline(stat['p95'], color='#777777', linestyle=':', linewidth=1.2,
			label='p95 %.4g' % stat['p95'])
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel('windows')
		ax.grid(axis='y', color='#dddddd', linewidth=0.6)
		ax.set_axisbelow(True)
		ax.legend(frameon=False, fontsize=9)
		ax.text(0.99, 0.97, 'n=%d\nmax=%.4g' % (len(values), stat['max']),
			transform=ax.transAxes, ha='right', va='top', fontsize=9)
		for side in ('top', 'right'):
			ax.spines[side].set_visible(False)
	fig.suptitle('MidiTranslator anomaly scan — 256 source lines extended to @tick')

	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	tmp = None
	try:
		with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(path)),
			prefix='.%s.' % os.path.basename(path), suffix='.png', delete=False) as f:
			tmp = f.name
		fig.savefig(tmp, dpi=130)
		os.replace(tmp, path)
	except Exception:
		if tmp and os.path.exists(tmp):
			os.unlink(tmp)
		raise
	finally:
		plt.close(fig)
	return True


def make_report (args, identity, corpus, windows, songs, errors, exhaustive, self_check_ok):
	ok = [row for row in windows if row['status'] == 'ok']
	metrics = [row['metrics'] for row in ok]
	loss_dist = distribution([row['loss'] for row in metrics])
	err_dist = distribution([row['err'] for row in metrics])
	all_song_coverage = len(songs) == corpus['shared_files'] and all(song['source_coverage_complete'] for song in songs)
	complete = exhaustive and not errors and all_song_coverage and len(ok) == len(windows)
	return {
		'schema': 'midi-translator-anomaly-report', 'version': 1,
		'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
		'status': 'complete' if complete else ('failed' if errors else 'limited'),
		'complete': complete, 'exhaustive': exhaustive, 'self_check': self_check_ok,
		'invocation': sys.argv, 'torch_version': torch.__version__, 'device': str(args.device),
		'run': identity, 'corpus': corpus,
		'windowing': {'nominal_source_lines': args.window_lines,
			'right_boundary': 'first source @tick line at or after nominal end; otherwise EOF',
			'random_crop': False, 'start_jitter': 0, 'resampling': False, 'truncation': False},
		'coverage': {
			'songs_expected': corpus['shared_files'], 'songs_attempted': len(songs),
			'songs_complete': sum(song['source_coverage_complete'] for song in songs),
			'windows_attempted': len(windows), 'windows_scored': len(ok),
			'all_source_covered': all_song_coverage,
		},
		'aggregate': aggregate_metrics(metrics),
		'distributions': {'loss': loss_dist, 'err': err_dist},
		'songs': songs, 'windows': windows, 'errors': errors,
	}


# --- synthetic checks ----------------------------------------------------------------------

def synthetic_file (n_lines, ticks):
	lines = ['event 0'] * n_lines
	for line, tick in ticks:
		lines[line] = '@tick %d' % tick
	return _File('\n'.join(lines), 'tick')


def self_check ():
	# Exact boundary, extension, a short final window and gap-free chaining.
	file = synthetic_file(600, [(0, 0), (256, 1), (520, 2)])
	windows = list(iter_source_windows(file, 256))
	assert [(w['start'], w['nominal_end'], w['end']) for w in windows] == [
		(0, 256, 256), (256, 512, 520), (520, 600, 600)]
	validate_tiling(file, windows)

	# Sparse marks can force a large extension; a markless file is still one honest whole-file crop.
	sparse = synthetic_file(700, [(0, 0), (610, 1)])
	sparse_windows = list(iter_source_windows(sparse, 256))
	assert sparse_windows[0]['end'] == 610
	assert sparse_windows[0]['end'] - sparse_windows[0]['nominal_end'] == 354
	validate_tiling(sparse, sparse_windows)
	header = synthetic_file(300, [(149, 0), (260, 1)])
	header_windows = list(iter_source_windows(header, 128))
	validate_tiling(header, header_windows)
	assert header_windows[0]['end'] == 260
	markless = synthetic_file(31, [])
	markless_windows = list(iter_source_windows(markless, 256))
	assert [(w['start'], w['end'], w['a'], w['z']) for w in markless_windows] == [(0, 31, 0, 0)]
	validate_tiling(markless, markless_windows)

	# Known predictions: two labels, exactly one error. Additive aggregation must weight by tokens.
	pred = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
	labels = torch.tensor([0, 0])
	metric = score_logits(pred, labels)
	assert metric['n_target'] == 2 and metric['n_error'] == 1 and metric['err'] == 0.5
	other = dict(loss_sum=3.0, loss=1.0, n_target=3, n_error=0, err=0.0)
	agg = aggregate_metrics([metric, other])
	assert agg['n_target'] == 5 and agg['n_error'] == 1 and abs(agg['err'] - 0.2) < 1e-12
	assert abs(agg['loss'] - (metric['loss_sum'] + 3.0) / 5) < 1e-12
	return True


# --- driver --------------------------------------------------------------------------------

def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--run', default=DEFAULT_RUN)
	ap.add_argument('--checkpoint', default=None, help='default: <run>/best.chkpt (no fallback)')
	ap.add_argument('--corpus', default=DEFAULT_CORPUS)
	ap.add_argument('--source-dir', default='midi-seq2-irregular')
	ap.add_argument('--target-dir', default='midi-seq2-score')
	ap.add_argument('--window-lines', type=int, default=256)
	ap.add_argument('--expect-files', type=int, default=100, help='required exact pair count; 0 disables')
	ap.add_argument('--device', default='cuda')
	ap.add_argument('--max-files', type=int, default=0, help='smoke limit; 0 = all')
	ap.add_argument('--max-windows', type=int, default=0, help='global smoke limit; 0 = all')
	ap.add_argument('--out-dir', default=DEFAULT_OUT)
	ap.add_argument('--report', default=None, help='JSON path; default: <out-dir>/report.json')
	ap.add_argument('--plot', default=None, help='PNG path; default: <out-dir>/distributions.png')
	ap.add_argument('--self-check-only', action='store_true')
	args = ap.parse_args()

	if args.window_lines < 1 or args.max_files < 0 or args.max_windows < 0:
		ap.error('window-lines must be positive and limits must be non-negative')
	if str(args.device).startswith('cuda') and not torch.cuda.is_available():
		ap.error('CUDA was requested but is not available; pass --device cpu')
	self_check_ok = self_check()
	print('synthetic self-check: PASS')
	if args.self_check_only:
		return 0

	args.device = torch.device(args.device)
	report_path = os.path.abspath(args.report or os.path.join(args.out_dir, 'report.json'))
	plot_path = os.path.abspath(args.plot or os.path.join(args.out_dir, 'distributions.png'))
	config, model, dataset, identity, corpus = load_run(args)
	print('checkpoint: %s (epoch %s)' % (identity['checkpoint'], identity['checkpoint_epoch']))
	print('corpus    : %d exact pairs, %s -> %s' % (
		corpus['shared_files'], args.source_dir, args.target_dir))

	names = dataset.names[:args.max_files or None]
	exhaustive = not args.max_files and not args.max_windows
	windows_out, songs, errors = [], [], []
	total_attempted = 0
	stop = False
	train_cap = int((config['data.args'] or {}).get('max_tokens') or 0)
	model_cap = int((config['model.args'] or {}).get('max_seq_len') or 0)
	wrapper_checked = False

	with torch.inference_mode():
		for song_index, name in enumerate(names):
			source = _get_file(dataset.source, dataset.arm_source, name, 'tick')
			target = _get_file(dataset.source, dataset.arm_target, name, 'tick')
			planned = list(iter_source_windows(source, args.window_lines))
			validate_tiling(source, planned)
			attempted = []
			previous_target_end = None
			for window_index, window in enumerate(planned):
				if args.max_windows and total_attempted >= args.max_windows:
					stop = True
					break
				total_attempted += 1
				a, z = window['a'], window['z']
				align = dataset._align(source, target, a, z)
				if align is None:
					error = ValueError('tick alignment produced an empty target range')
					failure = error_json('alignment', error, name, window_index)
					errors.append(failure)
					windows_out.append(dict(id='%s:w%04d' % (name.rsplit('.', 2)[0], window_index),
						status='error', song=name, song_index=song_index, window_index=window_index,
						source={'range': [window['start'], window['end']]}, error=failure))
					attempted.append(window)
					continue
				ids, sep, positions = dataset._assemble(source, target, a, z, align, 0)
				record = window_record(dataset, name, song_index, window_index, source, target,
					window, align, ids, sep, positions, previous_target_end, train_cap, model_cap)
				previous_target_end = align[1]
				attempted.append(window)
				try:
					metric, pred, labels = evaluate_window(dataset, model, ids, sep, positions)
					if not wrapper_checked:
						wrapper_loss = float(model._loss(pred, labels).item())
						if abs(wrapper_loss - metric['loss']) > 2e-6:
							raise AssertionError('direct CE disagrees with MidiTranslatorLoss._loss')
						wrapper_checked = True
					record['metrics'] = metric
					record['status'] = 'ok'
				except torch.OutOfMemoryError as error:
					failure = error_json('cuda_oom', error, name, window_index)
					record['status'], record['error'] = 'error', failure
					errors.append(failure)
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				except Exception as error:
					failure = error_json('inference', error, name, window_index)
					record['status'], record['error'] = 'error', failure
					errors.append(failure)
				windows_out.append(record)
				if total_attempted % 50 == 0:
					print('  windows %4d  songs %3d/%d  last T=%d' % (
						total_attempted, song_index + 1, len(names), len(ids)))

			song_rows = [row['metrics'] for row in windows_out
				if row.get('song') == name and row.get('status') == 'ok']
			coverage_complete = len(attempted) == len(planned) and (not attempted
				or attempted[0]['start'] == 0 and attempted[-1]['end'] == len(source.lines))
			songs.append(dict(song=name, song_index=song_index, source_lines=len(source.lines),
				target_lines=len(target.lines), windows_planned=len(planned), windows_attempted=len(attempted),
				windows_scored=len(song_rows), source_coverage_complete=coverage_complete,
				metrics=aggregate_metrics(song_rows)))
			if stop:
				break

	report = make_report(args, identity, corpus, windows_out, songs, errors,
		exhaustive, self_check_ok)
	plot_ok = plot_report(plot_path, windows_out, report['distributions'])
	report['plot'] = plot_path if plot_ok else None
	atomic_json(report_path, report)
	print('\nstatus     : %s' % report['status'])
	print('windows    : %d scored / %d attempted' % (
		report['coverage']['windows_scored'], report['coverage']['windows_attempted']))
	print('loss / err : %s / %s (token-weighted)' % (
		report['aggregate']['loss'], report['aggregate']['err']))
	print('report     : %s' % report_path)
	print('plot       : %s' % (plot_path if plot_ok else 'not written'))
	if errors:
		print('errors     : %d (see report)' % len(errors))
	return 0 if not errors else 1


if __name__ == '__main__':
	sys.exit(main())
