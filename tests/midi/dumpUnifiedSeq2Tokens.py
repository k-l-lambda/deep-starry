#!/usr/bin/env python3
'''Dump the current unified Lilylet/midiseq2 vocabulary as a readable table.

The mixed Seq2Seq2 mapping is SYNTHESIZED from assets/lilylet-tokenizer.json and
assets/midiseq2Vocab.yaml (never checked in as an asset), and a training run pins its own copy in
its config directory. That makes it awkward to eyeball, hence this dump: it prints every unified id
with the source-local id it came from, grouped by region, so a mapping change is reviewable as a
diff of two dumps.

By default it reads the CANONICAL mapping built from today's assets. Point --vocab at a run's
unifiedSeq2Vocab.json to dump what that run is actually pinned to instead.

Usage:
  python tests/midi/dumpUnifiedSeq2Tokens.py                       # -> tests/output/unifiedSeq2Vocab.txt
  python tests/midi/dumpUnifiedSeq2Tokens.py --vocab <run>/unifiedSeq2Vocab.json
  python tests/midi/dumpUnifiedSeq2Tokens.py -o /tmp/vocab.txt
'''

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from starry.midi.data.unifiedSeq2Tokenizer import (  # noqa: E402
	ARTIFACT_TYPE, ARTIFACT_VERSION, build_unified_vocab, load_unified_vocab)


DEFAULT_OUTPUT = os.path.join(REPO_ROOT, 'tests', 'output', 'unifiedSeq2Vocab.txt')
_REGIONS = ('special', 'lilylet', 'midiseq2')
_REGION_NOTES = {
	'special': 'shared controls + reserve — both modalities use these same ids',
	'lilylet': 'Lilylet content, remapped from source-local ids (byte values in the legacy tokenizer)',
	'midiseq2': 'midiseq2 content, remapped from source-local ids (no controls of its own)',
}


def show (token):
	'''Escape a token so control characters and whitespace survive a text dump.

	Lilylet content is largely raw bytes — tab, newline, backspace — which would otherwise break the
	table apart. Everything printable is shown verbatim so the common case stays readable.'''
	if token and token.isprintable() and not token.isspace():
		return token
	return token.encode('unicode_escape').decode('ascii')


def lines (artifact):
	blocks = artifact['blocks']
	entries = artifact['entries']
	out = [
		f'# unified Seq2Seq2 vocabulary: {artifact["type"]} v{artifact["version"]}',
		f'# vocab_size    {artifact["vocab_size"]}',
		f'# mapping_sha256 {artifact["mapping_sha256"]}',
		'#',
		'# provenance (digests of the CONTENT slices this mapping was built from):',
	]
	out.extend(f'#   {key:<24} {value}' for key, value in sorted(artifact['provenance'].items()))
	out.extend((
		'#',
		'# special ids: ' + '  '.join(f'{name}={value}'
			for name, value in sorted(artifact['special_ids'].items(), key=lambda kv: kv[1])),
		'#',
		'# blocks:',
	))
	out.extend(f'#   {name:<9} ids {blocks[name]["offset"]:>4}..'
		f'{blocks[name]["offset"] + blocks[name]["size"] - 1:<4} '
		f'size {blocks[name]["size"]:>4}  from source-local {blocks[name]["local_start"]}'
		for name in _REGIONS)

	for name in _REGIONS:
		block = blocks[name]
		out.extend(('', f'== {name}  [{_REGION_NOTES[name]}]',
			f'{"id":>5}  {"local":>5}  token'))
		for i in range(block['size']):
			entry = entries[block['offset'] + i]
			out.append(f'{entry["id"]:>5}  {entry["local_id"]:>5}  {show(entry["token"])}')
	return out


def main ():
	parser = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--vocab', help='a run-local unifiedSeq2Vocab.json to dump '
		'(default: build the canonical mapping from the current assets)')
	parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT)
	args = parser.parse_args()

	artifact = load_unified_vocab(args.vocab) if args.vocab else build_unified_vocab()
	os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
	with open(args.output, 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines(artifact)) + '\n')

	source = args.vocab or 'assets (canonical)'
	print(f'{ARTIFACT_TYPE} v{ARTIFACT_VERSION}: {artifact["vocab_size"]} tokens from {source}')
	print(f'mapping_sha256 {artifact["mapping_sha256"]}')
	print(f'written to {args.output}')


if __name__ == '__main__':
	main()
