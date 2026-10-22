#!/usr/bin/env python3
'''Packed vs loose InterleavedSeq2: the two layouts must produce identical samples and batches.

Builds, in a temp dir, the inputs tools/midi/packInterleavedLilylet.py expects from a loose mixed corpus
(default $DATA_DIR/test202608): a packMidiseq2Shards-style `<XX>.zip` set holding the midiseq2 arm,
`lyl/<XX>/<YY>/<id>.lyl`, and `measures-<XX>.json`. Runs the packer, then asserts:

  * the packed corpus holds exactly the pairs the loose feeder accepts (rejects counted by reason);
  * describe() agrees field for field on every sample, both directions, val (deterministic) crops;
  * collated batches are tensor-identical.

Usage: python tests/midi/interleaved_seq2_packed_check.py [--root DIR]
'''

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(REPO_ROOT)

import torch  # noqa: E402

from starry.midi.data.interleavedSeq2 import InterleavedSeq2  # noqa: E402
from starry.midi.data.seq2seq2 import _recursive_files  # noqa: E402
from starry.midi.data.unifiedSeq2Tokenizer import write_unified_vocab  # noqa: E402

ARM = 'midi-seq2-irregular'


def build_inputs (root, work):
	lyl = _recursive_files(os.path.join(root, 'lyl'), '.lyl')
	midi = _recursive_files(os.path.join(root, ARM), '.txt')
	with open(os.path.join(root, 'metadata', 'measures.json'), 'r', encoding='utf-8') as f:
		measures = json.load(f)
	ids = sorted(set(lyl) & set(midi))
	by_shard = defaultdict(list)
	for sid in ids:
		by_shard[sid[:2]].append(sid)

	packages = os.path.join(work, 'packages')
	os.makedirs(packages)
	for shard, members in by_shard.items():
		with zipfile.ZipFile(os.path.join(packages, f'{shard}.zip'), 'w') as z:
			for sid in members:
				z.write(midi[sid], f'{ARM}/{sid}.midiseq2.txt')
	with open(os.path.join(packages, 'index.json'), 'w') as f:
		json.dump(dict(arms=[ARM], shard_chars=2, samples=len(ids),
			shards={s: [f'{i}.midiseq2.txt' for i in m] for s, m in by_shard.items()}), f)

	for sid in ids:
		dest = os.path.join(work, 'lyl', sid[:2], sid[2:4])
		os.makedirs(dest, exist_ok=True)
		shutil.copy(lyl[sid], os.path.join(dest, sid + '.lyl'))
	os.makedirs(os.path.join(work, 'metadata'))
	for shard, members in by_shard.items():
		with open(os.path.join(work, 'metadata', f'measures-{shard}.json'), 'w') as f:
			json.dump({sid: measures[sid] for sid in members if sid in measures}, f)
	return ids


def feeder (root, vocab, direction, packed):
	src, tgt = direction
	dirs = {'midiseq2': ARM, 'lilylet': 'lyl'}
	return InterleavedSeq2(root, '0/1', source_format=src, target_format=tgt, source_dir=dirs[src],
		target_dir=dirs[tgt], measures_path='metadata/measures.json', vocab_path=vocab, line_range=(2, 12),
		max_tokens=2048, style_dropout=0.3, random_crop=False, packed=packed)


def main ():
	ap = argparse.ArgumentParser()
	ap.add_argument('--root', default=os.path.join(os.environ.get('DATA_DIR', ''), 'test202608'))
	args = ap.parse_args()

	work = tempfile.mkdtemp(prefix='interleaved-packed-check-')
	try:
		ids = build_inputs(args.root, work)
		out = os.path.join(work, 'packed')
		subprocess.run([sys.executable, os.path.join(REPO_ROOT, 'tools', 'midi', 'packInterleavedLilylet.py'),
			'--packages', os.path.join(work, 'packages'), '--arm', ARM, '--lyl', os.path.join(work, 'lyl'),
			'--measures', os.path.join(work, 'metadata'), '--out', out, '--workers', '4'],
			check=True, stdout=subprocess.DEVNULL)
		with open(os.path.join(out, 'index.json')) as f:
			manifest = json.load(f)
		vocab = os.path.join(work, 'vocab.json')
		write_unified_vocab(vocab)

		for direction in (('midiseq2', 'lilylet'), ('lilylet', 'midiseq2')):
			loose = feeder(args.root, vocab, direction, False)
			packed = feeder(out, vocab, direction, None)
			assert packed.packed and not loose.packed
			assert packed.names == loose.names, (len(packed.names), len(loose.names))
			for i in range(len(loose.names)):
				a, b = loose.describe(i), packed.describe(i)
				assert a == b, f'{direction} sample {i} ({a["name"]}) differs'
			ba = loose.collateBatch([loose[i] for i in range(len(loose))])
			bb = packed.collateBatch([packed[i] for i in range(len(packed))])
			assert all(torch.equal(ba[k], bb[k]) for k in ba)
			print('%-22s %d samples identical, batch %s' % ('%s -> %s' % direction, len(loose), tuple(ba['input_ids'].shape)))
		print('candidates %d | packed %d | rejects %s' % (len(ids), manifest['samples'], manifest['rejects']))
		assert manifest['samples'] + sum(manifest['rejects'].values()) == len(ids)
		print('interleaved_seq2_packed_check: OK')
	finally:
		shutil.rmtree(work)


if __name__ == '__main__':
	main()
