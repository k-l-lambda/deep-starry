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
import heapq
import json
import struct
import math
import os
import statistics
import sys
import tempfile
import traceback

import numpy as np
import torch
import torch.nn.functional as F

try:
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt					# noqa: E402
except ImportError:
	plt = None

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
		random_crop=False, seed=0, vocab_path=vocab_path, pos_style='sep', packed=None)
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


# --- streamed report and plot ----------------------------------------------------------------

class ScanWriter:
    """Bounded-memory trace writer, aggregate metrics, and top anomalies."""
    def __init__(self, out_dir, top_k=100):
        self.out_dir=os.path.abspath(out_dir); os.makedirs(self.out_dir, exist_ok=True)
        self.paths={k: os.path.join(self.out_dir, f'{k}.jsonl') for k in ('windows','songs','errors')}
        self.paths['metrics']=os.path.join(self.out_dir,'metrics.f64')
        self.files={k:open(v,'w',encoding='utf-8') for k,v in self.paths.items() if k!='metrics'}
        self.files['metrics']=open(self.paths['metrics'],'wb')
        self.top_k=top_k; self.top={'loss':[],'err':[]}; self.serial=0
        self.windows_attempted=self.windows_scored=self.songs_attempted=self.songs_complete=0
        self.n_target=self.n_error=0; self.loss_sum=0.0; self.error_count=0
    def _json(self, key, row):
        json.dump(row,self.files[key],ensure_ascii=False,allow_nan=False,separators=(',',':')); self.files[key].write('\n')
    def write_window(self,row):
        self.windows_attempted+=1; self._json('windows',row)
        if row.get('status')!='ok': return
        m=row['metrics']; self.windows_scored+=1; self.n_target+=int(m['n_target']); self.n_error+=int(m['n_error']); self.loss_sum+=float(m['loss_sum'])
        self.files['metrics'].write(struct.pack('<dd',float(m['loss']),float(m['err']))); self.serial+=1
        for key in ('loss','err'):
            item=(float(m[key]),self.serial,row); heap=self.top[key]
            if len(heap)<self.top_k: heapq.heappush(heap,item)
            elif item[0]>heap[0][0]: heapq.heapreplace(heap,item)
    def write_song(self,row):
        self.songs_attempted+=1; self.songs_complete+=int(bool(row.get('source_coverage_complete'))); self._json('songs',row)
    def write_error(self,row): self.error_count+=1; self._json('errors',row)
    def aggregate(self):
        return dict(windows=self.windows_scored,n_target=self.n_target,loss_sum=self.loss_sum,n_error=self.n_error,loss=self.loss_sum/self.n_target if self.n_target else None,err=self.n_error/self.n_target if self.n_target else None)
    def top_rows(self): return {k:[x[2] for x in sorted(v,reverse=True)] for k,v in self.top.items()}
    def flush(self):
        for f in self.files.values(): f.flush()
    def close(self):
        for f in self.files.values(): f.flush(); os.fsync(f.fileno()); f.close()

def trace_record(record):
    out={k:record[k] for k in ('id','status','song','song_index','window_index')}
    for group, keys in (('source',('range','lines','nominal_end','extension_lines','mark_range','head','tail','boundary')),('target',('range','lines','overlap_with_previous','gap_from_previous','alignment'))):
        if group in record: out[group]={k:record[group].get(k) for k in keys}
    if 'sequence' in record: out['sequence']=record['sequence']
    if 'metrics' in record: out['metrics']=record['metrics']
    if 'error' in record: out['error']=record['error']
    return out

def metric_values(path):
    values=np.fromfile(path,dtype='<f8')
    if values.size%2: raise ValueError('corrupt metrics sidecar: odd float count')
    return values.reshape(-1,2)

def distribution_array(values):
    if not len(values): return dict(count=0,min=None,mean=None,median=None,p90=None,p95=None,p99=None,max=None)
    q=np.percentile(values,[50,90,95,99])
    return dict(count=int(len(values)),min=float(np.min(values)),mean=float(np.mean(values)),median=float(q[0]),p90=float(q[1]),p95=float(q[2]),p99=float(q[3]),max=float(np.max(values)))

def plot_values(path, loss, err, summaries):
    if plt is None or not len(loss): return False
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,values,title,color,xlabel,stat in ((axes[0],loss,'Per-window cross-entropy','#4C78A8','loss',summaries['loss']),(axes[1],err,'Per-window token error rate','#E45756','err',summaries['err'])):
        bins=min(100,max(10,round(math.sqrt(len(values)))))
        if stat['min']==stat['max']: bins=1
        ax.hist(values,bins=bins,color=color,edgecolor='white',linewidth=.5); ax.axvline(stat['median'],color='#222',linestyle='--',label='median %.4g'%stat['median']); ax.axvline(stat['p95'],color='#777',linestyle=':',label='p95 %.4g'%stat['p95'])
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel('windows'); ax.grid(axis='y',color='#ddd',linewidth=.6); ax.set_axisbelow(True); ax.legend(frameon=False,fontsize=9); ax.text(.99,.97,'n=%d\nmax=%.4g'%(len(values),stat['max']),transform=ax.transAxes,ha='right',va='top',fontsize=9)
        for side in ('top','right'): ax.spines[side].set_visible(False)
    fig.suptitle('MidiTranslator anomaly scan — 256 source lines extended to @tick'); os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True); tmp=None
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(path)),prefix='.%s.'%os.path.basename(path),suffix='.png',delete=False) as f: tmp=f.name
        fig.savefig(tmp,dpi=130); os.replace(tmp,path)
    finally:
        if tmp and os.path.exists(tmp): os.unlink(tmp)
        plt.close(fig)
    return True

def sidecar_metadata(paths): return {k:{'path':os.path.abspath(v),'bytes':os.path.getsize(v),'sha256':sha256_file(v)} for k,v in paths.items()}

def make_stream_report(args,identity,corpus,writer,exhaustive,self_check_ok,distributions,plot_path):
    selected=int(corpus['selected_files']); all_covered=writer.songs_attempted==selected and writer.songs_complete==selected; complete=exhaustive and not writer.error_count and all_covered and writer.windows_attempted==writer.windows_scored
    return {'schema':'midi-translator-anomaly-report','version':2,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete' if complete else ('failed' if writer.error_count else 'limited'),'complete':complete,'exhaustive':exhaustive,'self_check':self_check_ok,'invocation':sys.argv,'torch_version':torch.__version__,'device':str(args.device),'run':identity,'corpus':corpus,'shard':{'index':args.shard_index,'count':args.num_shards},'windowing':{'nominal_source_lines':args.window_lines,'right_boundary':'first source @tick line at or after nominal end; otherwise EOF','random_crop':False,'start_jitter':0,'resampling':False,'truncation':False},'coverage':{'songs_corpus':corpus['shared_files'],'songs_expected':selected,'songs_attempted':writer.songs_attempted,'songs_complete':writer.songs_complete,'windows_attempted':writer.windows_attempted,'windows_scored':writer.windows_scored,'all_source_covered':all_covered},'aggregate':writer.aggregate(),'distributions':distributions,'top':writer.top_rows(),'errors':{'count':writer.error_count,'path':os.path.abspath(writer.paths['errors'])},'sidecars':sidecar_metadata(writer.paths),'plot':plot_path}

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

def evaluate_windows (dataset, model, items):
	batch = dataset.collateBatch([(torch.tensor(x['ids'],dtype=torch.long), x['sep'], torch.tensor(x['positions'],dtype=torch.long), 0) for x in items])
	for row,x in enumerate(items):
		ids,sep=x['ids'],x['sep']
		if int(batch['target_mask'][row,:sep+1].sum()) != 0: raise AssertionError('source or <sep> is supervised')
		if int(batch['target_mask'][row].sum()) != len(ids)-sep-1: raise AssertionError('target mask does not cover whole target')
		if ids[-1] != dataset.tokenizer.eos_id or not bool(batch['target_mask'][row,len(ids)-1]): raise AssertionError('<eos> is not final supervised token')
	logits=model.deducer(batch['input_ids'],batch['masks'],batch['position_ids']); pred,labels=model._shift(batch,logits)
	counts=batch['target_mask'][:,1:].sum(dim=1).tolist(); out=[]; offset=0
	for count in counts:
		count=int(count); out.append(score_logits(pred[offset:offset+count],labels[offset:offset+count])); offset+=count
	if offset != len(labels): raise AssertionError('batched target segmentation disagrees with _shift')
	return out,pred,labels

def main ():
	ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--run',default=DEFAULT_RUN); ap.add_argument('--checkpoint',default=None); ap.add_argument('--corpus',default=DEFAULT_CORPUS)
	ap.add_argument('--source-dir',default='midi-seq2-irregular'); ap.add_argument('--target-dir',default='midi-seq2-score'); ap.add_argument('--window-lines',type=int,default=256)
	ap.add_argument('--expect-files',type=int,default=100,help='required exact pair count; 0 disables'); ap.add_argument('--device',default='cuda'); ap.add_argument('--batch-size',type=int,default=1)
	ap.add_argument('--max-files',type=int,default=0); ap.add_argument('--max-windows',type=int,default=0); ap.add_argument('--shard-index',type=int,default=0); ap.add_argument('--num-shards',type=int,default=1)
	ap.add_argument('--top-k',type=int,default=100); ap.add_argument('--progress-every-songs',type=int,default=100); ap.add_argument('--out-dir',default=DEFAULT_OUT)
	ap.add_argument('--report',default=None); ap.add_argument('--plot',default=None); ap.add_argument('--self-check-only',action='store_true'); args=ap.parse_args()
	if args.window_lines<1 or args.batch_size<1 or min(args.max_files,args.max_windows)<0: ap.error('window-lines/batch-size positive; limits non-negative')
	if args.num_shards<1 or not 0<=args.shard_index<args.num_shards: ap.error('invalid shard selection')
	if args.top_k<1 or args.progress_every_songs<1: ap.error('top-k/progress positive')
	if str(args.device).startswith('cuda') and not torch.cuda.is_available(): ap.error('CUDA unavailable; pass --device cpu')
	self_check_ok=self_check(); print('synthetic self-check: PASS',flush=True)
	if args.self_check_only: return 0
	args.device=torch.device(args.device); report_path=os.path.abspath(args.report or os.path.join(args.out_dir,'report.json')); plot_path=os.path.abspath(args.plot or os.path.join(args.out_dir,'distributions.png'))
	config,model,dataset,identity,corpus=load_run(args); all_names=dataset.names; selected=all_names[args.shard_index::args.num_shards]
	if args.max_files: selected=selected[:args.max_files]
	manifest=os.path.join(args.corpus,'index.json'); corpus.update(layout=type(dataset.source).__name__,manifest=os.path.abspath(manifest) if os.path.isfile(manifest) else None,manifest_sha256=sha256_file(manifest) if os.path.isfile(manifest) else None,selected_files=len(selected))
	print('checkpoint: %s (epoch %s)'%(identity['checkpoint'],identity['checkpoint_epoch']),flush=True); print('corpus: %d pairs, %s, shard %d/%d (%d songs)'%(corpus['shared_files'],corpus['layout'],args.shard_index,args.num_shards,len(selected)),flush=True)
	writer=ScanWriter(args.out_dir,args.top_k); exhaustive=not args.max_files and not args.max_windows; total=0; stop=False; wrapper_checked=False; train_cap=int((config['data.args'] or {}).get('max_tokens') or 0); model_cap=int((config['model.args'] or {}).get('max_seq_len') or 0)
	def failure(kind,error,name=None,index=None):
		row=error_json(kind,error,name,index); writer.write_error(row); return row
	def score_pending(pending,song_metrics):
		nonlocal wrapper_checked
		if not pending:return
		try: metrics,pred,labels=evaluate_windows(dataset,model,pending)
		except Exception as error:
			if len(pending)>1:
				for item in pending: score_pending([item],song_metrics)
				return
			item=pending[0]; row=failure('cuda_oom' if isinstance(error,torch.OutOfMemoryError) else 'inference',error,item['name'],item['window_index']); item['record']['status']='error'; item['record']['error']=row; writer.write_window(trace_record(item['record']))
			if isinstance(error,torch.OutOfMemoryError) and torch.cuda.is_available(): torch.cuda.empty_cache()
			return
		if not wrapper_checked:
			weighted=sum(x['loss_sum'] for x in metrics)/sum(x['n_target'] for x in metrics)
			if abs(float(model._loss(pred,labels).item())-weighted)>2e-6: raise AssertionError('direct CE disagrees with wrapper loss')
			wrapper_checked=True
		for item,metric in zip(pending,metrics): item['record'].update(metrics=metric,status='ok'); writer.write_window(trace_record(item['record'])); song_metrics.append(metric)
	try:
		with torch.inference_mode():
			for local_index, name in enumerate(selected):
				idx = args.shard_index + local_index * args.num_shards
				pending, attempted, song_metrics = [], [], []
				try:
					source = _get_file(dataset.source, dataset.arm_source, name, 'tick')
					target = _get_file(dataset.source, dataset.arm_target, name, 'tick')
					planned = list(iter_source_windows(source, args.window_lines))
					validate_tiling(source, planned)
				except Exception as error:
					row = failure('song_preparation', error, name)
					writer.write_song({'song': name, 'song_index': idx, 'status': 'error', 'source_coverage_complete': False, 'metrics': aggregate_metrics([]), 'error': row})
					continue
				previous_target_end = None
				for wi, window in enumerate(planned):
					if args.max_windows and total >= args.max_windows:
						stop = True
						break
					total += 1
					attempted.append(window)
					a, z = window['a'], window['z']
					try:
						align = dataset._align(source, target, a, z)
						if align is None:
							raise ValueError('tick alignment produced an empty target range')
						ids, sep, positions = dataset._assemble(source, target, a, z, align, 0)
						record = window_record(dataset, name, idx, wi, source, target, window, align, ids, sep, positions, previous_target_end, train_cap, model_cap)
						previous_target_end = align[1]
					except Exception as error:
						row = failure('window_preparation', error, name, wi)
						writer.write_window({'id': '%s:w%04d' % (name.rsplit('.', 2)[0], wi), 'status': 'error', 'song': name, 'song_index': idx, 'window_index': wi, 'source': {'range': [window['start'], window['end']]}, 'error': row})
						continue
					pending.append({'name': name, 'window_index': wi, 'ids': ids, 'sep': sep, 'positions': positions, 'record': record})
					if len(pending) >= args.batch_size:
						score_pending(pending, song_metrics)
						pending = []
				score_pending(pending, song_metrics)
				covered = len(attempted) == len(planned) and (not attempted or (attempted[0]['start'] == 0 and attempted[-1]['end'] == len(source.lines)))
				writer.write_song({'song': name, 'song_index': idx, 'status': 'ok' if covered else 'limited', 'source_lines': len(source.lines), 'target_lines': len(target.lines), 'windows_planned': len(planned), 'windows_attempted': len(attempted), 'windows_scored': len(song_metrics), 'source_coverage_complete': covered, 'metrics': aggregate_metrics(song_metrics)})
				if writer.songs_attempted % args.progress_every_songs == 0:
					writer.flush()
					print('songs %d/%d windows %d loss %.5g err %.5g' % (writer.songs_attempted, len(selected), writer.windows_scored, writer.aggregate()['loss'] or 0, writer.aggregate()['err'] or 0), flush=True)
				if stop:
					break
	finally: writer.close()
	values=metric_values(writer.paths['metrics']); distributions={'loss':distribution_array(values[:,0]),'err':distribution_array(values[:,1])}; plot_ok=plot_values(plot_path,values[:,0],values[:,1],distributions); report=make_stream_report(args,identity,corpus,writer,exhaustive,self_check_ok,distributions,plot_path if plot_ok else None); atomic_json(report_path,report)
	print('\nstatus: %s'%report['status']); print('windows: %d scored / %d attempted'%(report['coverage']['windows_scored'],report['coverage']['windows_attempted'])); print('loss / err: %s / %s'%(report['aggregate']['loss'],report['aggregate']['err'])); print('report: %s'%report_path); print('plot: %s'%(plot_path if plot_ok else 'not written')); return 0 if not writer.error_count else 1


if __name__ == '__main__':
	sys.exit(main())
