"""Reproducible full-file greedy vs guarded beam comparison without score truth.

Input fidelity and output hygiene are proxies, not transcription accuracy. The
evaluation's independent greedy pass is the unmodified baseline algorithm, evaluated
with identical model/window settings. Sample membership is fixed before inference.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/midi'), str(ROOT / 'tests/midi')]
import torch
import translateMidiseq2 as T
from sequenceBeamTranslator import SequenceBeamTranslator
from translate_accuracy_check import parse_notes, pairing_defects, f1

RUN = '/home/camus/data/models/deep-starry-logs/midi/20260909-midi-translator-nota1m0909-lines360-l16d256'
MODEL = TOKENIZER = CONFIG = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lcs_length(a, b):
    """Exact pitch-sequence LCS length, using Python integer bitsets."""
    masks = {}
    for i, value in enumerate(a):
        masks[value] = masks.get(value, 0) | (1 << i)
    state = 0
    for value in b:
        x = state | masks.get(value, 0)
        state = x & ~(x - ((state << 1) | 1))
    return state.bit_count()


def content_metrics(lines, source):
    notes, starts, _ = parse_notes(lines)
    source_notes, _, _ = parse_notes(source)
    pitches, source_pitches = [n[1] for n in notes], [n[1] for n in source_notes]
    overlap = f1(Counter(pitches), Counter(source_pitches))
    lcs = lcs_length(source_pitches, pitches)
    denom = len(pitches) + len(source_pitches)
    on, reon, orphan, stuck = pairing_defects(lines)
    bar_pitches = defaultdict(list)
    for tick, pitch, channel, bar in notes:
        bar_pitches[bar].append(pitch)
    longest = run = 0
    previous = None
    for bar in sorted(bar_pitches):
        signature = tuple(bar_pitches[bar])
        run = run + 1 if signature == previous else 1
        longest = max(longest, run)
        previous = signature
    # These grid rates describe notation regularity, not correct rhythm. Both
    # straight and triplet subdivisions are reported, without rewarding either.
    grid_rates = {str(grid): sum(abs(t / grid - round(t / grid)) < 1e-7
                                 for t, *_ in notes) / max(1, len(notes)) for grid in [60, 40]}
    return dict(notes=len(notes), source_notes=len(source_notes),
                note_ratio=len(notes) / max(1, len(source_notes)),
                input_pitch_precision=overlap[0], input_pitch_recall=overlap[1], input_pitch_f1=overlap[2],
                input_pitch_lcs=lcs, input_lcs_precision=lcs / max(1, len(pitches)),
                input_lcs_recall=lcs / max(1, len(source_pitches)),
                input_lcs_f1=2 * lcs / denom if denom else 1.,
                re_on=reon, orphan_off=orphan, stuck=stuck,
                pairing_defects_per_100_notes=100 * (reon + orphan + stuck) / max(1, on),
                bars=len(starts), max_identical_pitch_bar_run=longest,
                last_onset_tick=notes[-1][0] if notes else 0,
                onset_grid_fraction=grid_rates)


def initialize(args):
    global MODEL, TOKENIZER, CONFIG
    torch.set_num_threads(args['threads'])
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    CONFIG, MODEL = T.load_model(args['run'], args['checkpoint'], args['device'])
    TOKENIZER, _ = T.resolve_tokenizer(args['run'], CONFIG)
    if CONFIG['model.type'] != 'MidiTranslator':
        raise ValueError('this comparison harness expects the requested decoder-only checkpoint')


def one_file(item, args):
    name, source_sha = item['file'], item['sha256']
    out = Path(args['out'])
    result_path = out / 'results' / (name + '.json')
    if result_path.exists():
        row = json.loads(result_path.read_text())
        if row.get('status') == 'ok':
            return row
    path = Path(args['corpus']) / 'midiseq2' / name
    if digest(path) != source_sha:
        raise ValueError('input changed after sampling: ' + name)
    source = path.read_text().splitlines()
    T.assert_midiseq2(source, str(path), TOKENIZER)
    data = CONFIG['data.args'] or {}
    tr = SequenceBeamTranslator(MODEL, TOKENIZER, pos_style=data.get('pos_style', 'flat'),
        src_window=640, prime_window=320, max_token=2048, device=args['device'],
        source_eom=bool(data.get('source_eom')), beam_size=4, branch_k=4,
        length_alpha=.7, logprob_margin=2., alignment_weight=0.,
        rescue_alignment_weight=.5, guard=True, quality_stop=True, align_advance=True)
    baseline = T.SlidingTranslator(MODEL, TOKENIZER, pos_style=data.get('pos_style', 'flat'),
        src_window=640, prime_window=320, max_token=2048, device=args['device'],
        source_eom=bool(data.get('source_eom')))
    greedy_ids, greedy_stats = baseline.translate(source, max_steps=0)
    start = time.monotonic()
    ids, stats = tr.translate(source, max_steps=0)
    elapsed = time.monotonic() - start
    streams = [('greedy', greedy_ids, greedy_stats),
               ('beam', ids, stats), ('raw_candidate', tr.candidate_output, tr.candidate_stats)]
    if tr.rescue_output is not None:
        streams.append(('rescue', tr.rescue_output, tr.rescue_stats))
    row = dict(file=name, source_sha256=source_sha, status='ok', seconds=elapsed,
               source=content_metrics(source, source), streams={},
               runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                            device=torch.cuda.get_device_name() if args['device'].startswith('cuda') else 'cpu'))
    (out / 'input' / name).write_text('\n'.join(source) + '\n')
    for label, output_ids, output_stats in streams:
        body = T.render_lines(output_ids, TOKENIZER, tr.keywords)
        (out / label / name).write_text('\n'.join(body) + '\n')
        row['streams'][label] = dict(content_metrics(body, source), stats=output_stats,
                                     output_sha256=digest(out / label / name))
    # Save completed pairs atomically; never resume half a comparison as success.
    tmp = result_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2) + '\n')
    tmp.replace(result_path)
    return row


def summarize(rows):
    good = [r for r in rows if r.get('status') == 'ok']
    metrics = ['input_pitch_f1', 'input_lcs_f1', 'input_pitch_recall', 'input_lcs_recall',
               'input_pitch_precision', 'note_ratio', 'pairing_defects_per_100_notes']
    result = dict(requested=len(rows), evaluated=len(good), errors=[r for r in rows if r.get('status') != 'ok'],
                  warning='No ground truth: input-fidelity and hygiene proxies only.', modes={})
    for mode in ['greedy', 'beam', 'raw_candidate']:
        rr = [r['streams'][mode] for r in good]
        if rr:
            result['modes'][mode] = dict(files=len(rr), completed=sum(r['stats']['done'] for r in rr),
                means={k:statistics.mean(r[k] for r in rr) for k in metrics},
                seconds_total=sum(r['stats']['elapsed'] for r in rr),
                seconds_mean=statistics.mean(r['stats']['elapsed'] for r in rr),
                early_stops=sum(bool(r['stats'].get('early_stop')) for r in rr))
    result['selection'] = dict(Counter(r['streams']['beam']['stats']['guard']['selected'] for r in good))
    result['paired'] = {}
    for k in ['input_pitch_f1', 'input_lcs_f1', 'pairing_defects_per_100_notes']:
        ds = [(r['file'], r['streams']['beam'][k] - r['streams']['greedy'][k]) for r in good]
        result['paired'][k] = dict(increased=sum(d>1e-9 for _,d in ds),
                                  equal=sum(abs(d)<=1e-9 for _,d in ds),
                                  decreased=sum(d< -1e-9 for _,d in ds), deltas=ds)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--checkpoint', default='/tmp/beam-redesign/checkpoint.chkpt')
    ap.add_argument('--corpus', default='/home/camus/data/midi/yt-piano')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample-count', type=int, default=20)
    ap.add_argument('--sample-seed', type=int, default=20260923)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--prepare-only', action='store_true')
    args = vars(ap.parse_args())
    out = Path(args['out'])
    for d in ['', 'input', 'greedy', 'beam', 'raw_candidate', 'rescue', 'results']:
        (out / d).mkdir(parents=True, exist_ok=True)
    files = sorted((Path(args['corpus']) / 'midiseq2').glob('*.txt'))
    sample = sorted(random.Random(args['sample_seed']).sample(files, args['sample_count']))
    items = [dict(file=p.name, sha256=digest(p), notes=sum(line.startswith('note_on ') for line in p.read_text().splitlines())) for p in sample]
    code = ['tools/midi/translateMidiseq2.py', 'tools/midi/translateMidiseq2Beam.py',
            'tools/midi/sequenceBeamTranslator.py', 'starry/midi/sequenceBeam.py',
            'starry/midi/translationControl.py', 'starry/midi/align.py',
            'tests/midi/yt_piano_compare.py', 'tests/midi/translate_accuracy_check.py']
    meta = dict(args={k:v for k,v in args.items() if k != 'prepare_only'},
                population=len(files), files=items, checkpoint_sha256=digest(args['checkpoint']),
                code_sha256={p:digest(ROOT / p) for p in code},
                settings=dict(beam=4,branch_k=4,src_window=640,prime_window=320,max_token=2048,
                              max_steps=0,rank='auto',align_advance=True,quality_stop=True,
                              length_alpha=.7,logprob_margin=2.,rescue_alignment_weight=.5,
                              precision='float32; TF32 disabled'), ground_truth=None)
    mp = out / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != meta:
        raise ValueError('checkpoint, sample, settings or code changed; use a new output directory')
    mp.write_text(json.dumps(meta,indent=2)+'\n')
    print('SAMPLE',len(items),'/',len(files),'notes',sum(i['notes'] for i in items),flush=True)
    if args['prepare_only']:
        return
    started = time.monotonic()
    rows = []
    # Longest first keeps both workers occupied through the full-song comparison.
    tasks = sorted(items, key=lambda i:-i['notes'])
    with ProcessPoolExecutor(max_workers=args['workers'], mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(args,)) as pool:
        pending = {pool.submit(one_file, item, args):item for item in tasks}
        for future in as_completed(pending):
            item = pending[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(file=item['file'], status='error', error=repr(exc))
                (out/'results'/(item['file']+'.error.json')).write_text(json.dumps(row,indent=2))
            rows.append(row)
            if row['status']=='ok':
                g,b = row['streams']['greedy'],row['streams']['beam']
                print(f'{len(rows):2d}/20 {row["file"]} pitch {g["input_pitch_f1"]:.4f}->{b["input_pitch_f1"]:.4f} '
                      f'LCS {g["input_lcs_f1"]:.4f}->{b["input_lcs_f1"]:.4f} '
                      f'done {g["stats"]["done"]}->{b["stats"]["done"]} '
                      f'selected={b["stats"]["guard"]["selected"]} {row["seconds"]:.1f}s',flush=True)
            else:
                print('ERROR',row,flush=True)
            summary=summarize(rows)
            summary['wall_seconds']=time.monotonic()-started
            (out/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows.sort(key=lambda r:r['file'])
    summary=summarize(rows);summary['wall_seconds']=time.monotonic()-started
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
