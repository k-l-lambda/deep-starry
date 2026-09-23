"""Paired greedy/beam evaluation against a fixed reference, including missing output.

Development and holdout indices are disjoint, fixed before evaluation. Each file
is scored in full (or as the explicitly requested first N bars). No short output
is silently dropped, and reference coverage never follows the generated length.
Pin --checkpoint to a copy: a mutable best.chkpt is not a reproducible experiment.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/midi'), str(ROOT / 'tests/midi')]

import torch
import translateMidiseq2 as T
from sequenceBeamTranslator import SequenceBeamTranslator, SequenceBeamEncDecTranslator
from translate_accuracy_check import parse_notes, quantize, f1, pairing_defects


SPLITS = dict(dev=[0, 12, 25, 37, 50, 62, 75, 87],
              holdout=[4, 11, 18, 32, 39, 46, 57, 64, 71, 82, 89, 96],
              confirm=[2, 9, 16, 23, 30, 44, 53, 60, 67, 78, 85, 94])


def read_excerpt(path, bars):
    lines = path.read_text().splitlines()
    if not bars:
        return lines
    out = []
    for line in lines:
        if line.startswith('@measure ') and int(line.split()[1]) > bars:
            break
        if line == 'end_of_track':
            break
        out.append(line)
    return out + ['end_of_track']


def score_output(output, reference, grid=120):
    predicted, pstarts, _ = parse_notes(output)
    expected, rstarts, _ = parse_notes(reference)
    def counter(notes, starts, kind):
        if kind == 'pitch':
            return Counter(p for t, p, c, m in notes)
        if kind == 'relative':
            return Counter((m, quantize(t - starts[m], grid), p) for t, p, c, m in notes)
        scale = 1 if kind == 'exact_onset' else grid
        return Counter((quantize(t, scale), p) for t, p, c, m in notes)
    scores = {}
    for kind in ('pitch', 'onset', 'relative', 'exact_onset'):
        values = f1(counter(predicted, pstarts, kind), counter(expected, rstarts, kind))
        scores.update({kind + '_' + key: value for key, value in zip(('p', 'r', 'f1'), values)})
    scores.update(notes_out=len(predicted), notes_ref=len(expected),
                  bars_out=len(pstarts), bars_ref=len(rstarts),
                  pairing_defects=pairing_defects(output), reference_pairing_defects=pairing_defects(reference))
    return scores


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['greedy', 'lm', 'align', 'auto'], default='auto')
    ap.add_argument('--split', choices=['dev', 'holdout', 'confirm', 'all'], default='confirm')
    ap.add_argument('--indices', help='explicit comma-separated sorted file indices')
    ap.add_argument('--bars', type=int, default=8, help='first N reference/source bars; 0 = whole files')
    ap.add_argument('--grid', type=int, default=120)
    ap.add_argument('--beam', type=int, default=4)
    ap.add_argument('--alignment-weight', type=float, default=.5)
    ap.add_argument('--length-alpha', type=float, default=.7)
    ap.add_argument('--logprob-margin', type=float, default=2.)
    ap.add_argument('--src-window', type=int, default=640)
    ap.add_argument('--prime-window', type=int, default=320)
    ap.add_argument('--max-token', type=int, default=2048)
    ap.add_argument('--max-steps', type=int, default=16)
    ap.add_argument('--align-advance', action='store_true')
    ap.add_argument('--no-guard', action='store_true')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    if args.bars < 0 or args.grid < 1:
        ap.error('--bars must be nonnegative and --grid positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    corpus = Path(args.corpus)
    names = sorted(p.name for p in (corpus / 'midiseq2-irregular').glob('*.txt')
                   if (corpus / 'midiseq2-score' / p.name).exists())
    indices = (list(map(int, args.indices.split(','))) if args.indices else
               list(range(len(names))) if args.split == 'all' else SPLITS[args.split])
    if not names or any(i < 0 or i >= len(names) for i in indices):
        ap.error('no corpus pairs, or selected index outside corpus')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    code = ['starry/midi/sequenceBeam.py', 'starry/midi/align.py',
            'tools/midi/sequenceBeamTranslator.py', 'tools/midi/translateMidiseq2.py',
            'tests/midi/beam_quality_check.py', 'tests/midi/translate_accuracy_check.py']
    meta = dict(args=vars(args), checkpoint_sha256=digest(args.checkpoint),
                code_sha256={name: digest(ROOT / name) for name in code},
                files=[names[i] for i in indices])
    meta_path = out / 'meta.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError('existing output has different checkpoint/code/settings; use a new directory')
    meta_path.write_text(json.dumps(meta, indent=2))
    config, model = T.load_model(args.run, args.checkpoint, args.device)
    tokenizer, _ = T.resolve_tokenizer(args.run, config)
    data = config['data.args'] or {}
    encdec = config['model.type'] == 'MidiTranslatorEncDec'
    if encdec:
        model.sep_id, model.eos_id, model.pad_id = tokenizer.sep_id, tokenizer.eos_id, tokenizer.pad_id
    cls = (T.SlidingEncDecTranslator if encdec else T.SlidingTranslator) if args.mode == 'greedy' else (
           SequenceBeamEncDecTranslator if encdec else SequenceBeamTranslator)
    rows = []
    for i in indices:
        name = names[i]
        result_path = out / (name + '.json')
        if result_path.exists():
            row = json.loads(result_path.read_text())
        else:
            source = read_excerpt(corpus / 'midiseq2-irregular' / name, args.bars)
            reference = read_excerpt(corpus / 'midiseq2-score' / name, args.bars)
            options = dict(pos_style=data.get('pos_style', 'flat'), src_window=args.src_window,
                           prime_window=args.prime_window, max_token=args.max_token, device=args.device,
                           source_eom=bool(data.get('source_eom')), align_advance=args.align_advance)
            if args.mode != 'greedy':
                options.update(beam_size=args.beam, alignment_weight=(args.alignment_weight
                               if args.mode == 'align' else 0.), length_alpha=args.length_alpha,
                               logprob_margin=args.logprob_margin, guard=not args.no_guard,
                               rescue_alignment_weight=(args.alignment_weight if args.mode == 'auto' else 0.))
            translator = cls(model, tokenizer, **options)
            start = time.monotonic()
            ids, stats = translator.translate(source, max_steps=args.max_steps)
            body = T.render_lines(ids, tokenizer, translator.keywords)
            row = score_output(body, reference, args.grid)
            row.update(index=i, file=name, seconds=time.monotonic() - start, stats=stats)
            if args.mode != 'greedy':
                row['search'] = translator.search_report
                if translator.greedy_output is not None:
                    for label, output_ids, output_stats in [
                        ('greedy', translator.greedy_output, translator.greedy_stats),
                        ('candidate', translator.candidate_output, translator.candidate_stats)]:
                        candidate_body = T.render_lines(output_ids, tokenizer, translator.keywords)
                        row[label] = dict(score_output(candidate_body, reference, args.grid), stats=output_stats)
                        (out / (name + '.' + label)).write_text('\n'.join(candidate_body) + '\n')
                    if translator.rescue_output is not None:
                        rescue_body = T.render_lines(translator.rescue_output, tokenizer, translator.keywords)
                        row['rescue'] = dict(score_output(rescue_body, reference, args.grid),
                                            stats=translator.rescue_stats)
                        (out / (name + '.rescue')).write_text('\n'.join(rescue_body) + '\n')
            result_path.write_text(json.dumps(row, indent=2))
            (out / name).write_text('\n'.join(body) + '\n')
        rows.append(row)
        print(f'{i:3d} {name[:12]} onset={row["onset_f1"]:.4f} pitch={row["pitch_f1"]:.4f} '
              f'relative={row["relative_f1"]:.4f} done={row["stats"]["done"]} '
              f'{row["seconds"]:.1f}s', flush=True)
    summary = {key: statistics.mean(r[key] for r in rows)
               for key in ('onset_f1', 'pitch_f1', 'relative_f1', 'exact_onset_f1', 'seconds')}
    summary.update(files=len(rows), completed=sum(r['stats']['done'] for r in rows),
                   failures=[r['file'] for r in rows if not r['stats']['done']])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
