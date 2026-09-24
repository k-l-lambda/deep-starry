"""Regenerate the seven historical Greedy fallbacks using beam-only auto.

Keep the original evaluation immutable. Persist complete candidate text, stop
reasons and hashes; publishing to tests/output is a separate verified copy step.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import yt_piano_compare as C
from sequenceBeamTranslator import SequenceBeamTranslator


def one(item, args):
    out = Path(args['out'])
    name = item['file']
    target = out / 'results' / (name + '.json')
    if target.exists():
        return json.loads(target.read_text())
    path = Path(args['corpus']) / 'midiseq2' / name
    assert C.digest(path) == item['sha256'], name
    source = path.read_text().splitlines()
    C.T.assert_midiseq2(source, str(path), C.TOKENIZER)
    data = C.CONFIG['data.args'] or {}
    tr = SequenceBeamTranslator(C.MODEL, C.TOKENIZER, pos_style=data.get('pos_style', 'flat'),
        src_window=640, prime_window=320, max_token=2048, device='cuda',
        source_eom=bool(data.get('source_eom')), beam_size=4, branch_k=4,
        length_alpha=.7, logprob_margin=2., alignment_weight=0.,
        rescue_alignment_weight=.5, guard=True, quality_stop=True, align_advance=True)
    started = time.monotonic()
    ids, stats = tr.translate(source, max_steps=0)
    assert tr.greedy_output is None and tr.greedy_stats is None
    assert stats['guard']['selected'] in ('beam', 'alignment')
    row = dict(file=name, status='ok', source_sha256=item['sha256'],
               seconds=time.monotonic()-started, streams={}, source=C.content_metrics(source, source))
    streams = [('beam', ids, stats), ('raw_candidate', tr.candidate_output, tr.candidate_stats)]
    if tr.rescue_output is not None:
        streams.append(('rescue', tr.rescue_output, tr.rescue_stats))
    shutil.copy2(path, out / 'input' / name)
    for mode, tokens, st in streams:
        lines = C.T.render_lines(tokens, C.TOKENIZER, tr.keywords)
        dest = out / mode / name
        dest.write_text('\n'.join(lines)+'\n')
        row['streams'][mode] = dict(C.content_metrics(lines, source), stats=st,
                                   output_sha256=C.digest(dest))
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(row, indent=2)+'\n')
    tmp.replace(target)
    return row


def initialize(args):
    if not C.torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    C.initialize(args)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='/tmp/yt-piano-beam20-no-greedy-20260924')
    ap.add_argument('--prepare-only', action='store_true')
    a=ap.parse_args()
    old_path=C.ROOT/'docs/midi/yt-piano-beam20-results.json'
    old=json.loads(old_path.read_text())
    args=dict(old['manifest']['args'], out=a.out)
    selected={r['file'] for r in old['rows'] if r['streams']['beam']['stats']['guard']['selected']=='greedy'}
    assert len(selected)==7
    items=[i for i in old['manifest']['files'] if i['file'] in selected]
    assert C.digest(args['checkpoint'])==old['manifest']['checkpoint_sha256']
    code=list(old['manifest']['code_sha256'])+['tests/midi/yt_piano_regenerate.py']
    manifest=dict(args=args, files=items, selection='historical auto selected greedy',
                  original_results_sha256=C.digest(old_path),
                  checkpoint_sha256=C.digest(args['checkpoint']),
                  settings=dict(old['manifest']['settings'], selection_policy='beam_only',
                                greedy_execution=False, incomplete_retry=True,
                                accept_retry='done and no early_stop and pitch F1 >= primary - 0.05'),
                  code_sha256={p:C.digest(C.ROOT/p) for p in code})
    out=Path(a.out)
    for mode in ['input','beam','raw_candidate','rescue','results','previous_auto']:
        (out/mode).mkdir(parents=True,exist_ok=True)
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:
        raise ValueError('Resume manifest mismatch')
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    for p,h in manifest['code_sha256'].items():
        dest=out/'code-snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(C.ROOT/p,dest)
    old_rows={r['file']:r for r in old['rows']}
    for item in items:
        name=item['file'];previous=Path(old['manifest']['args']['out'])/'beam'/name
        assert C.digest(previous)==old_rows[name]['streams']['beam']['output_sha256']
        shutil.copy2(previous,out/'previous_auto'/name)
    print('FIXED',len(items),'files; no Greedy execution; CUDA required',flush=True)
    if a.prepare_only:return
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize,initargs=(args,)) as pool:
        futures={pool.submit(one,item,args):item for item in sorted(items,key=lambda x:-x['notes'])}
        for future in as_completed(futures):
            item=futures[future]
            try:row=future.result()
            except Exception as e:
                row=dict(file=item['file'],status='error',error=repr(e))
                (out/'results'/(item['file']+'.error.json')).write_text(json.dumps(row,indent=2))
            rows.append(row)
            if row['status']=='ok':
                s=row['streams']['beam'];print(len(rows),'/7',row['file'],
                    s['stats']['guard']['selected'],'done',s['stats']['done'],
                    'stop',s['stats'].get('early_stop'),'bars',s['bars'],
                    'seconds',round(row['seconds'],1),flush=True)
            else:print('ERROR',row,flush=True)
            (out/'progress.json').write_text(json.dumps(dict(completed=len(rows),
                errors=[r for r in rows if r['status']!='ok'],wall_seconds=time.monotonic()-started),indent=2))
    result=dict(manifest=manifest,rows=sorted(rows,key=lambda r:r['file']),wall_seconds=time.monotonic()-started)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if any(r['status']!='ok' for r in rows):raise RuntimeError('Some files failed; see results')
    print('FINISHED 7/7',flush=True)


if __name__=='__main__':main()
