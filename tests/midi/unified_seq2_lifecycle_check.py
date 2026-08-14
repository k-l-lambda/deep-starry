#!/usr/bin/env python3
'''Focused checks for the mixed Seq2Seq2 run-local vocabulary lifecycle.'''

import copy
import json
import os
import shutil
import sys
import tempfile

import yaml

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from starry.midi.data.seq2CondPachifier import Midiseq2Tokenizer  # noqa: E402
from starry.midi.data.unifiedSeq2Tokenizer import (  # noqa: E402
	UnifiedSeq2Tokenizer, build_unified_vocab, load_unified_vocab, write_unified_vocab)
from starry.midi.models.midiTranslator import MidiTranslatorLoss  # noqa: E402
from starry.utils.config import Configuration  # noqa: E402


def raises (error, function):
	try:
		function()
	except error:
		return True
	return False


def mixed_state ():
	return {
		'id': 'mixed-test',
		'env': None,
		'data': {'type': 'Seq2Seq2', 'args': {
			'source_format': 'midiseq2', 'target_format': 'lilylet'}},
		'model': {'type': 'MidiTranslator', 'args': {}},
	}


def main ():
	artifact = build_unified_vocab()
	assert artifact['vocab_size'] == 1094
	assert artifact['blocks']['lilylet'] == {'offset': 0, 'size': 256}
	assert artifact['blocks']['midiseq2'] == {'offset': 256, 'size': 838}
	tokenizer = UnifiedSeq2Tokenizer(artifact=artifact)
	assert tokenizer.tokens[4] == '<mask>'
	assert tokenizer.tokens[tokenizer.midiseq2_eom_id] == '<eom>'
	assert tokenizer.lilylet_mask_id != tokenizer.midiseq2_eom_id

	root = tempfile.mkdtemp(prefix='unified-seq2-check-')
	try:
		run = os.path.join(root, 'run')
		os.makedirs(run)
		config = Configuration(run, mixed_state())
		path = os.path.join(run, 'unifiedSeq2Vocab.json')
		assert os.path.isfile(path)
		assert load_unified_vocab(path)['mapping_sha256'] == artifact['mapping_sha256']
		assert config['data.args.vocab_path'] == path
		assert config['model.args.vocab_path'] == path
		assert config['model.args.vocab_size'] == 1094

		state_path = os.path.join(run, '.state.yaml')
		for _ in range(2):
			config.save()
			state = yaml.safe_load(open(state_path, 'r'))
			assert state['_unified_vocab'] == 'unifiedSeq2Vocab.json'
			assert state['data']['args']['vocab_path'] == 'unifiedSeq2Vocab.json'
			assert state['model']['args']['vocab_path'] == 'unifiedSeq2Vocab.json'

		moved = os.path.join(root, 'moved')
		shutil.copytree(run, moved)
		resumed = Configuration(moved)
		moved_path = os.path.join(moved, 'unifiedSeq2Vocab.json')
		assert resumed['data.args.vocab_path'] == moved_path
		assert resumed['model.args.vocab_path'] == moved_path

		missing = os.path.join(root, 'missing')
		shutil.copytree(run, missing)
		os.remove(os.path.join(missing, 'unifiedSeq2Vocab.json'))
		assert raises(ValueError, lambda: Configuration(missing))

		corrupt = os.path.join(root, 'corrupt')
		shutil.copytree(run, corrupt)
		with open(os.path.join(corrupt, 'unifiedSeq2Vocab.json'), 'w') as f:
			f.write('{}')
		assert raises(ValueError, lambda: Configuration(corrupt))

		bad_digest = copy.deepcopy(artifact)
		bad_digest['mapping_sha256'] = '0' * 64
		bad_path = os.path.join(root, 'bad.json')
		with open(bad_path, 'w') as f:
			json.dump(bad_digest, f)
		assert raises(ValueError, lambda: load_unified_vocab(bad_path))

		wrong_size = mixed_state()
		wrong_size['model']['args']['vocab_size'] = 838
		wrong_run = os.path.join(root, 'wrong-size')
		os.makedirs(wrong_run)
		assert raises(ValueError, lambda: Configuration(wrong_run, wrong_size))

		volatile = os.path.join(root, 'volatile')
		os.makedirs(volatile)
		assert raises(ValueError, lambda: Configuration(volatile, mixed_state(), volatile=True))
		assert os.listdir(volatile) == []

		roundtrip = os.path.join(root, 'roundtrip.json')
		write_unified_vocab(roundtrip, artifact)
		assert load_unified_vocab(roundtrip) == artifact
	finally:
		shutil.rmtree(root)

	print('unified Seq2Seq2 lifecycle checks passed')


if __name__ == '__main__':
	main()
