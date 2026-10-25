#!/usr/bin/env python3
'''Does translateMidiseq2ToLilylet.bar_units cut a source into the units InterleavedSeq2 trained on?

Reads the loose mixed corpus (default $DATA_DIR/test202608). For every file the feeder's own
`_encode_unit` gives the training source unit of each played measure; `bar_units` must reproduce them
from the annotated text under two placements of the bar line:

  corpus            the file as written: `@measure` before the elapse run leading into the bar.
                    Must match on EVERY bar -- this is the feeder's own convention.
  annotate_source   each `@measure` moved to just before its bar's first note_on, which is where
                    translateMidiseq2.annotate_source writes it. `bar_units` moves the elapse line in
                    front of the bar back across; what it cannot recover is reported (note_offs a
                    sustain deferred past the bar line), and must stay rare. MEASURED on test202608:
                    7694/7750 units exact. The bar COUNT differs on 17/99 files, every one of them a
                    piece ending in note-less bars: `bar_units` folds a trailing unit with no event
                    line into the bar before it, which is right for annotate_source's terminal
                    `@measure` and costs these closing rests.

No model and no GPU: this checks the text plumbing between the two stages, not translation quality.

Usage: DATA_DIR=~/data/midi python tests/midi/translate_midiseq2_to_lilylet_check.py [--root DIR]
'''

import argparse
import os
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from starry.midi.data.interleavedSeq2 import InterleavedSeq2  # noqa: E402
from starry.midi.data.unifiedSeq2Tokenizer import write_unified_vocab  # noqa: E402
from tools.midi.translateMidiseq2ToLilylet import bar_units  # noqa: E402


def annotate_placement (lines):
	'''The corpus text with each `@measure N` (N > 1) moved to just before its bar's first note_on.

	A bar with no note_on keeps its directive where the next one would go, so it stays an empty bar,
	as annotate_source writes one.
	'''
	out, pending = [], []
	for line in lines:
		if line.startswith('@measure') and line.split()[1] != '1':
			pending.append(line)
			continue
		if pending and line.startswith('note_on'):
			out.extend(pending)
			pending = []
		out.append(line)
	return out + pending


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--root', default=os.path.join(os.environ.get('DATA_DIR', ''), 'test202608'))
	args = ap.parse_args()

	with tempfile.TemporaryDirectory() as work:
		vocab = os.path.join(work, 'vocab.json')
		write_unified_vocab(vocab)
		ds = InterleavedSeq2(args.root, '0/1', source_dir='midi-seq2-irregular', target_dir='lyl',
			measures_path='metadata/measures.json', vocab_path=vocab)
		counts = dict(files=0, bars=0, corpus=0, annotate=0, annotate_files=0)
		for sid in ds.names:
			text = ds._mixed_text('midiseq2', sid).splitlines()
			mapping = ds._mixed_mapping(sid)
			_, lyl = ds._mixed_lilylet_document(sid)
			midi = ds._mixed_midi(sid)
			truth = [ds._encode_unit(n, lyl, midi, mapping)[0] for n in range(1, len(mapping) + 1)]
			counts['files'] += 1
			counts['bars'] += len(truth)
			for key, source in (('corpus', text), ('annotate', annotate_placement(text))):
				units = bar_units(source)
				got = [ds._mixed_encode_midi(body, False) for _, body in units]
				assert [n for n, _ in units] == list(range(1, len(units) + 1)), f'{sid}: bar numbers not contiguous'
				if key == 'corpus':
					assert got == truth, f'{sid}: corpus-placed bar lines do not give the training units'
				else:
					counts['annotate_files'] += len(got) == len(truth)
				counts[key] += sum(a == b for a, b in zip(got, truth))
		print('%d files, %d bars' % (counts['files'], counts['bars']))
		print('  corpus placement           units exact %d/%d' % (counts['corpus'], counts['bars']))
		print('  annotate_source placement  units exact %d/%d (%.3f), bar count right on %d/%d files' % (
			counts['annotate'], counts['bars'], counts['annotate'] / counts['bars'], counts['annotate_files'], counts['files']))
		assert counts['annotate'] / counts['bars'] > 0.95, 'the elapse-line move no longer recovers the training cut'
		print('translate_midiseq2_to_lilylet_check: OK')


if __name__ == '__main__':
	main()
