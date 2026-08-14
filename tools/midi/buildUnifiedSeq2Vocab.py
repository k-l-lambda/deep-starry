#!/usr/bin/env python3
'''Build a unified Lilylet/midiseq2 vocabulary for inspection or an explicit run artifact.'''

import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from starry.midi.data.unifiedSeq2Tokenizer import build_unified_vocab, write_unified_vocab


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output', required=True,
		help='explicit output path (training runs normally create this through Configuration)')
	args = parser.parse_args()
	output = os.path.abspath(args.output)
	assets = os.path.join(ROOT, 'assets') + os.sep
	if output.startswith(assets):
		raise ValueError('generated unified vocabularies must not be written under assets/')
	artifact = build_unified_vocab()
	write_unified_vocab(output, artifact)
	print(json.dumps({'output': output, 'vocab_size': artifact['vocab_size'],
		'mapping_sha256': artifact['mapping_sha256']}))


if __name__ == '__main__':
	main()
