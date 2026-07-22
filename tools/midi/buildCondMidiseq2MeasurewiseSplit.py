'''Pack the conditioned-MIDI measurewise dataset, midiseq2 SPLIT-MODALITY layout.

Sibling of tools/midi/buildCondMidiseq2Measurewise.py: same two-stage flow and same inputs
(dataset.yaml + lyl/ + midi-seq2/), but writes the SPLIT artifact for
starry.midi.data.seq2CondSplitPatchy — two separate patch tensors (lyl @ lyl_patch_size, midi @
patch_size) at their native widths and vocabularies, instead of one joint [T,64] tensor.

Two-stage flow:
  1. intelli-piano:  midi-txt/*.txt --midiTxtToSeq2.ts-->  midi-seq2/*.txt   (basic -> midiseq2)
  2. here:           dataset.yaml + lyl/ + midi-seq2/  -->  cond-midiseq2-split.lmmw.pt

Usage:
  python3 tools/midi/buildCondMidiseq2MeasurewiseSplit.py \
    --dataset ~/data/lilylet/midi-measurewise/test20260629/dataset.yaml \
    --out     ~/data/lilylet/midi-measurewise/test20260629/cond-midiseq2-split.lmmw.pt
  # (lyl-root / midi-seq2-root default to <dataset dir>/lyl and <dataset dir>/midi-seq2)
'''

import argparse
import os
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.lilylet.data.patchifier import LilyletTokenizer
from starry.midi.data.seq2CondSplitPachifier import Midiseq2Tokenizer, pack_split, PATCH_SIZE, LYL_PATCH_SIZE


def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--dataset', required=True, help='dataset.yaml path')
	ap.add_argument('--out', required=True, help='output artifact path (.pt index; shard sits beside it)')
	ap.add_argument('--lyl-root', default=None, help='dir of .lyl files (default: <dataset dir>/lyl)')
	ap.add_argument('--midi-seq2-root', default=None, help='dir of midiseq2 .txt (default: <dataset dir>/midi-seq2)')
	ap.add_argument('--lyl-tokenizer', default=os.path.join(REPO_ROOT, 'assets', 'lilylet-tokenizer.json'))
	ap.add_argument('--seq2-vocab', default=os.path.join(REPO_ROOT, 'assets', 'midiseq2Vocab.yaml'))
	ap.add_argument('--patch-size', type=int, default=PATCH_SIZE, help='midiseq2 measure-patch width (default 64)')
	ap.add_argument('--lyl-patch-size', type=int, default=LYL_PATCH_SIZE, help='lilylet encoder patch width (default 16)')
	ap.add_argument('--no-patch-stream', action='store_true', help='disable the [r:i/..] lilylet stream tags')
	ap.add_argument('--limit', type=int, default=0, help='pack only the first N samples (0 = all)')
	args = ap.parse_args()

	dataset = os.path.expanduser(args.dataset)
	ddir = os.path.dirname(os.path.abspath(dataset))
	lyl_root = os.path.expanduser(args.lyl_root) if args.lyl_root else os.path.join(ddir, 'lyl')
	midi_seq2_root = os.path.expanduser(args.midi_seq2_root) if args.midi_seq2_root else os.path.join(ddir, 'midi-seq2')
	out_path = os.path.expanduser(args.out)

	with open(dataset, encoding='utf-8') as f:
		Loader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
		d = yaml.load(f, Loader=Loader)
	samples = d['samples']
	if args.limit and args.limit > 0:
		samples = samples[:args.limit]

	lyl_tokenizer = LilyletTokenizer(os.path.expanduser(args.lyl_tokenizer))
	seq2_tokenizer = Midiseq2Tokenizer(os.path.expanduser(args.seq2_vocab))

	print(f'packing {len(samples)} samples (SPLIT layout)')
	print(f'  lyl-root       : {lyl_root}')
	print(f'  midi-seq2-root : {midi_seq2_root}')
	print(f'  out            : {out_path}')
	print(f'  midi patch_size {args.patch_size} | lyl_patch_size {args.lyl_patch_size} | seq2 vocab {seq2_tokenizer.vocab_size}')
	index = pack_split(samples, lyl_root, midi_seq2_root, lyl_tokenizer, seq2_tokenizer, out_path,
		patch_size=args.patch_size, lyl_patch_size=args.lyl_patch_size,
		patch_stream=not args.no_patch_stream)

	st = index['stats']
	print(f"packed {st['packed']}/{st['samples']}  skipped {st['skipped']}")
	for sid, why in st['skips'][:20]:
		print(f'  SKIP {sid[:50]}: {why}')


if __name__ == '__main__':
	main()
