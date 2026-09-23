"""Translate irregular MIDI with an additive, cached beam search.

The search shares window advancement with translateMidiseq2.py. It ranks log
probability minus a bounded optional per-note alignment penalty. Structural tokens
remain normal candidates; --beam 1 uses greedy generation. Inspection records the
search without changing its candidates or budget. --legacy reproduces the old
align-first translator and its command-line options.

Default auto mode compares an independent greedy output with the LM beam using
completion and input-pitch fidelity. Only a rejected beam triggers an alignment
retry; greedy remains the fallback. Score-reference labels are never consulted.
"""

import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from starry.utils.config import Configuration
from translateMidiseq2 import (resolve_checkpoint, resolve_tokenizer, load_model,
                              render_lines, compose_output, write_output, report_output, source_header)
from tools.midi.midiseq2Text import assert_midiseq2
from sequenceBeamTranslator import SequenceBeamTranslator, SequenceBeamEncDecTranslator

BeamTranslator = SequenceBeamTranslator
BeamEncDecTranslator = SequenceBeamEncDecTranslator


def main():
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy')
        from translateMidiseq2BeamLegacy import main as legacy_main
        return legacy_main()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='/home/camus/data/models/deep-starry-logs/midi/'
                    '20260909-midi-translator-nota1m0909-lines360-l16d256')
    ap.add_argument('--checkpoint')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output')
    ap.add_argument('--beam', type=int, default=4)
    ap.add_argument('--branch-k', type=int, default=4)
    ap.add_argument('--rank', choices=['auto', 'lm', 'align'], default='auto',
                    help='auto: LM beam, soft-alignment rescue on guard rejection, then greedy fallback')
    ap.add_argument('--alignment-weight', type=float, default=0.5,
                    help='nonnegative event penalty weight; zero is pure LM search')
    ap.add_argument('--length-alpha', type=float, default=0.7)
    ap.add_argument('--logprob-margin', type=float, default=2.0,
                    help='search only model top-k candidates within this many nats')
    ap.add_argument('--src-window', type=int, default=640)
    ap.add_argument('--prime-window', type=int, default=320)
    ap.add_argument('--advance-tokens', type=int, default=1)
    ap.add_argument('--max-token', type=int, default=2048)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--no-prime', action='store_true')
    ap.add_argument('--no-kv-cache', action='store_true')
    ap.add_argument('--no-guard', action='store_true',
                    help='disable the independent greedy check and alignment rescue')
    ap.add_argument('--align-advance', action='store_true',
                    help='use the same retired-output alignment as the greedy translator')
    ap.add_argument('--inspect', action='store_true', help='record all search positions; no budget change')
    ap.add_argument('--inspect-json')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    if args.beam < 1 or args.branch_k < 1:
        ap.error('--beam and --branch-k must be positive')
    for key in ('alignment_weight', 'length_alpha', 'logprob_margin'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            ap.error(f'--{key.replace("_", "-")} must be finite and nonnegative')
    if args.src_window < 1 or args.max_token < 2 or args.prime_window < 0 or args.max_steps < 0:
        ap.error('invalid window or generation budget')
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = Configuration.createOrLoad(args.run, volatile=True)
    checkpoint = resolve_checkpoint(args.run, config, args.checkpoint)
    config, model = load_model(args.run, checkpoint, args.device)
    tokenizer, vocab = resolve_tokenizer(args.run, config)
    data_args = config['data.args'] or {}
    pos_style = data_args.get('pos_style', 'flat')
    model_type = config['model.type']
    if model_type not in ('MidiTranslator', 'MidiTranslatorEncDec'):
        ap.error(f'unsupported model type {model_type!r}')
    if pos_style not in ('flat', 'sep', 'absolute'):
        ap.error(f'unsupported position style {pos_style!r}')
    cls = SequenceBeamTranslator
    if model_type == 'MidiTranslatorEncDec':
        cls = SequenceBeamEncDecTranslator
        model.sep_id, model.eos_id, model.pad_id = tokenizer.sep_id, tokenizer.eos_id, tokenizer.pad_id
    with open(args.input, encoding='utf-8') as f:
        lines = f.read().splitlines()
    assert_midiseq2(lines, args.input, tokenizer)
    weight = args.alignment_weight if args.rank == 'align' else 0.0
    rescue_weight = args.alignment_weight if args.rank == 'auto' else 0.0
    translator = cls(model, tokenizer, pos_style=pos_style, src_window=args.src_window,
                     max_token=args.max_token, device=args.device, prime=not args.no_prime,
                     source_eom=bool(data_args.get('source_eom')), advance_tokens=args.advance_tokens,
                     prime_window=args.prime_window, kv_cache=not args.no_kv_cache,
                     align_advance=args.align_advance, beam_size=args.beam, branch_k=args.branch_k,
                     alignment_weight=weight, length_alpha=args.length_alpha,
                     logprob_margin=args.logprob_margin, guard=not args.no_guard,
                     rescue_alignment_weight=rescue_weight,
                     inspect=args.inspect or bool(args.inspect_json))
    print(f'[search] additive beam={args.beam}, rank={args.rank}, alignment_weight={weight:g}, '
          f'length_alpha={args.length_alpha:g}, margin={args.logprob_margin:g}; '
          f'shared greedy window advancement; checkpoint={checkpoint}')
    ids, stats = translator.translate(lines, max_steps=args.max_steps, verbose=args.verbose)
    body = render_lines(ids, tokenizer, translator.keywords)
    out_path = args.output or os.path.join(REPO_ROOT, 'tests', 'output', 'translate_midiseq2',
                                         os.path.splitext(os.path.basename(args.input))[0] + '.beam.txt')
    write_output(out_path, compose_output(body, source_header(lines)))
    report_output(body, stats)
    if 'guard' in stats:
        guard = stats['guard']
        print(f'[guard] selected {guard["selected"]}; reason={guard["reason"]}; '
              f'input pitch F1 greedy={guard["greedy_source_f1"]:.4f}, '
              f'beam={guard["beam_source_f1"]:.4f}; total {stats["elapsed"]:.1f}s')
    print(f'[out] wrote {out_path}; end_of_track={stats["done"]}')
    if translator.inspect:
        path = args.inspect_json or os.path.splitext(out_path)[0] + '.search.json'
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(format='sequence-beam-v2', args=vars(args), checkpoint=checkpoint,
                           vocab=vocab, stats=stats, search=translator.search_report,
                           windows=translator.search_windows,
                           rescue_search=translator.rescue_search_report,
                           rescue_windows=translator.rescue_windows), f)
        print(f'[inspect] wrote {path} (sequence-beam-v2)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
