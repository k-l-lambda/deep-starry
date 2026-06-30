'''Pack the lilylet<->MIDI measurewise dataset into a conditioned-MIDI joint-sequence artifact.

Reads a dataset.yaml (produced by intelli-piano/tools/buildMeasurewiseDataset.ts) plus the
per-sample lilylet `.lyl` files and the whole-song MidiText `.txt` files (produced by
intelli-piano/tools/midiToTextWhole.ts), and writes ONE sharded artifact consumed by
starry.midi.data.condPatchy.CondMidiPatchy.

Each output item is a single joint sequence: lilylet patches (the score, read-only
condition) followed by midi patches (events bucketed into measures, each terminated by an
<eom> patch). See starry.midi.data.condPatchifier for the layout.

Usage:
  python3 tools/midi/buildCondMidiMeasurewise.py \
    --dataset ~/data/lilylet/midi-measurewise/test20260629/dataset.yaml \
    --out     ~/data/lilylet/midi-measurewise/test20260629/cond-midi.lmmw.pt
  # (lyl-root / midi-txt-root default to <dataset dir>/lyl and <dataset dir>/midi-txt)
'''

import argparse
import os
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.lilylet.data.patchifier import LilyletTokenizer
from starry.midi.tokenizer import MidiTokenizer
from starry.midi.data.condPatchifier import pack, PATCH_SIZE


def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--dataset', required=True, help='dataset.yaml path')
	ap.add_argument('--out', required=True, help='output artifact path (.pt index; shard sits beside it)')
	ap.add_argument('--lyl-root', default=None, help='dir of .lyl files (default: <dataset dir>/lyl)')
	ap.add_argument('--midi-txt-root', default=None, help='dir of whole-song MidiText .txt (default: <dataset dir>/midi-txt)')
	ap.add_argument('--lyl-tokenizer', default=os.path.join(REPO_ROOT, 'assets', 'lilylet-tokenizer.json'))
	ap.add_argument('--patch-size', type=int, default=PATCH_SIZE)
	ap.add_argument('--no-patch-stream', action='store_true', help='disable the [r:i/..] lilylet stream tags')
	ap.add_argument('--limit', type=int, default=0, help='pack only the first N samples (0 = all)')
	args = ap.parse_args()

	dataset = os.path.expanduser(args.dataset)
	ddir = os.path.dirname(os.path.abspath(dataset))
	lyl_root = os.path.expanduser(args.lyl_root) if args.lyl_root else os.path.join(ddir, 'lyl')
	midi_txt_root = os.path.expanduser(args.midi_txt_root) if args.midi_txt_root else os.path.join(ddir, 'midi-txt')
	out_path = os.path.expanduser(args.out)

	with open(dataset, encoding='utf-8') as f:
		d = yaml.safe_load(f)
	samples = d['samples']
	if args.limit and args.limit > 0:
		samples = samples[:args.limit]

	lyl_tokenizer = LilyletTokenizer(os.path.expanduser(args.lyl_tokenizer))
	midi_tokenizer = MidiTokenizer(args.patch_size)

	print(f'packing {len(samples)} samples')
	print(f'  lyl-root      : {lyl_root}')
	print(f'  midi-txt-root : {midi_txt_root}')
	print(f'  out           : {out_path}')
	index = pack(samples, lyl_root, midi_txt_root, lyl_tokenizer, midi_tokenizer, out_path,
		patch_size=args.patch_size, patch_stream=not args.no_patch_stream)

	st = index['stats']
	print(f"packed {st['packed']}/{st['samples']}  skipped {st['skipped']}")
	for sid, why in st['skips'][:20]:
		print(f'  SKIP {sid[:50]}: {why}')


if __name__ == '__main__':
	main()
