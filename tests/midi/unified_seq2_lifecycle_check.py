#!/usr/bin/env python3
'''Focused checks for the mixed Seq2Seq2 run-local vocabulary lifecycle.'''

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

import yaml
import torch

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(REPO_ROOT)

from starry.lilylet.data.patchifier import LilyletTokenizer  # noqa: E402
from starry.midi.data.seq2CondPachifier import (  # noqa: E402
	_ASSET_VOCAB as MIDI_ASSET, Midiseq2Tokenizer)
from starry.midi.data.unifiedSeq2Tokenizer import (  # noqa: E402
	LILYLET_ASSET, UnifiedSeq2Tokenizer, build_unified_vocab, load_unified_vocab,
	write_unified_vocab)
from starry.midi.models.midiTranslator import MidiTranslatorLoss  # noqa: E402
from starry.utils.config import Configuration  # noqa: E402

sys.path.append(os.path.join(REPO_ROOT, 'tools', 'midi'))
from translateMidiseq2 import resolve_tokenizer  # noqa: E402


def raises (error, function):
	try:
		function()
	except error:
		return True
	return False


def mixed_state (assets=({'type': 'UnifiedSeq2Vocab'},)):
	'''A mixed run's state. The run-local vocabulary is requested EXPLICITLY through `assets:` —
	Configuration no longer infers it from data.type/source_format.'''
	return {
		'id': 'mixed-test',
		'env': None,
		'imports': ['starry.midi.data.unifiedSeq2Tokenizer'],
		'assets': [dict(spec) for spec in assets],
		'data': {'type': 'Seq2Seq2', 'args': {
			'source_format': 'midiseq2', 'target_format': 'lilylet'}},
		'model': {'type': 'MidiTranslator', 'args': {}},
	}


def vocab_layout (artifact):
	'''The merged v3 mapping: one 16-id control region, then content-only modality blocks.'''
	assert artifact['type'] == 'merged-lilylet-midiseq2'
	assert artifact['version'] == 3
	assert artifact['vocab_size'] == 1094
	assert artifact['blocks'] == {
		'special': {'offset': 0, 'size': 16, 'local_start': 0},
		'lilylet': {'offset': 16, 'size': 248, 'local_start': 8},
		'midiseq2': {'offset': 264, 'size': 830, 'local_start': 8},
	}
	assert artifact['special_ids'] == {
		'pad': 0, 'bos': 1, 'eos': 2, 'unknown': 3, 'mask': 4, 'sep': 5, 'eom': 6}
	# The mapping is deterministic, so the digest pins it against silent asset drift.
	assert build_unified_vocab()['mapping_sha256'] == artifact['mapping_sha256']

	tok = UnifiedSeq2Tokenizer(artifact=artifact)
	assert tok.tokens[:7] == ['<pad>', '<bos>', '<eos>', '<unknown>', '<mask>', '<sep>', '<eom>']
	assert tok.tokens[7:16] == [f'<reserved_{i}>' for i in range(7, 16)]
	# No modality-specific control duplicates survive: <bos>/<eos>/<eom> are single ids.
	assert not any(token.startswith('<') and token.endswith('>') for token in tok.tokens[16:])
	assert (tok.eom_id, tok.mask_id, tok.sep_id) == (6, 4, 5)

	# Source-local remaps. Lilylet locals are BYTE values in the legacy tokenizer, so they must land
	# in the content block rather than staying where they were.
	assert (tok.lilylet_id(8), tok.lilylet_id(255)) == (16, 263)
	assert tok.lilylet_id(10) == 18						# newline, structurally significant
	assert (tok.midi_id(8), tok.midi_id(837)) == (264, 1093)
	assert tok.midi_id(4) == tok.eom_id					# MIDI <eom> folds into the shared control
	assert tok.lilylet_id(4) == tok.mask_id				# Lilylet local 4 is <mask>, not <eom>
	for local in (1, 2, 5):
		assert tok.lilylet_id(local) == tok.midi_id(local) == local
	# The source reserves have no unified counterpart, so a stray one raises instead of aliasing.
	for local in (6, 7):
		assert raises(ValueError, lambda local=local: tok.lilylet_id(local))
		assert raises(ValueError, lambda local=local: tok.midi_id(local))
	assert raises(ValueError, lambda: tok.lilylet_id(256))

	# Content strings shared by both modalities stay DISTINCT ids: the blocks are disjoint.
	shared = set(tok.lilylet_id_by_token) & set(tok.midiseq2_id_by_token)
	assert {'0', '9', 'a', 'f', '-', '_'} <= shared
	assert all(tok.lilylet_id_by_token[t] != tok.midiseq2_id_by_token[t] for t in shared)
	assert all(16 <= tok.lilylet_id_by_token[t] < 264 for t in shared)
	assert all(264 <= tok.midiseq2_id_by_token[t] < 1094 for t in shared)

	# Every id a real Lilylet encode can emit stays inside the control region or the Lilylet block.
	lyl_tok = LilyletTokenizer(LILYLET_ASSET)
	sample = '\\key c \\major \\time 4/4 c4 d e f |\ng4 a b c\' |\n'
	local_ids = lyl_tok.encode(sample)
	unified = tok.lilylet_ids(local_ids)
	assert len(unified) == len(local_ids) and local_ids
	assert all(i < 16 or 16 <= i < 264 for i in unified)
	assert max(local_ids) >= 8 and max(unified) >= 16
	assert unified == [tok.lilylet_id(i) for i in local_ids]
	return tok


def midiseq2_state ():
	'''A plain midiseq2 -> midiseq2 run pinning a COPY of the authoritative vocabulary.'''
	return {
		'id': 'midiseq2-test',
		'env': None,
		'imports': ['starry.midi.data.seq2CondPachifier'],
		'assets': [{'type': 'Midiseq2Vocab'}],
		'data': {'type': 'Seq2Seq2', 'args': {
			'source_format': 'midiseq2', 'target_format': 'midiseq2'}},
		'model': {'type': 'MidiTranslator', 'args': {}},
	}


def midiseq2_asset (root):
	'''The second asset type: a verbatim copy, sharing the base class's lifecycle with the mixed one.'''
	run = os.path.join(root, 'midiseq2-run')
	os.makedirs(run)
	config = Configuration(run, midiseq2_state())
	path = os.path.join(run, 'midiseq2Vocab.yaml')
	# Copied byte for byte, so the pinned file parses exactly as the asset does.
	assert open(path, 'rb').read() == open(MIDI_ASSET, 'rb').read()
	assert config['model.args.vocab_size'] == Midiseq2Tokenizer().vocab_size == 838
	assert config['model.args.eos_id'] == 2
	assert config['data.args.vocab_path'] == config['model.args.vocab_path'] == path

	config.save()
	state = yaml.safe_load(open(os.path.join(run, '.state.yaml'), 'r'))
	assert state['_assets'] == {'Midiseq2Vocab': 'midiseq2Vocab.yaml'}
	assert state['model']['args']['vocab_path'] == 'midiseq2Vocab.yaml'
	config.load()
	assert config['model.args.vocab_path'] == path

	moved = os.path.join(root, 'midiseq2-moved')
	shutil.copytree(run, moved)
	assert Configuration(moved)['model.args.vocab_path'] == os.path.join(moved, 'midiseq2Vocab.yaml')

	# A truncated pin drops the special block the module constants assume, so it must not resume.
	truncated = os.path.join(root, 'midiseq2-truncated')
	shutil.copytree(run, truncated)
	with open(os.path.join(truncated, 'midiseq2Vocab.yaml'), 'w') as f:
		f.write('vocab:\n  - <pad>\n')
	assert raises(Exception, lambda: Configuration(truncated))

	wrong = midiseq2_state()
	wrong['model']['args']['vocab_size'] = 1094
	wrong_run = os.path.join(root, 'midiseq2-wrong-size')
	os.makedirs(wrong_run)
	assert raises(ValueError, lambda: Configuration(wrong_run, wrong))

	# Volatile creation publishes the copy to scratch, leaving the named directory alone.
	volatile_run = os.path.join(root, 'midiseq2-volatile')
	os.makedirs(volatile_run)
	volatile_config = Configuration(volatile_run, midiseq2_state(), volatile=True)
	assert os.listdir(volatile_run) == []
	assert Midiseq2Tokenizer(volatile_config['model.args.vocab_path']).vocab_size == 838

	# The tool reads the RUN's copy, not today's asset: it is the only thing tying ids to weights.
	tokenizer, resolved = resolve_tokenizer(run, config)
	assert resolved == path and tokenizer.vocab_size == 838
	# A mixed run's pin is refused rather than rendered as if it were midiseq2.
	mixed_run = os.path.join(root, 'run')
	assert raises(ValueError, lambda: resolve_tokenizer(mixed_run, Configuration(mixed_run)))


def main ():
	artifact = build_unified_vocab()
	tokenizer = vocab_layout(artifact)

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
		assert config['model.args.eos_id'] == tokenizer.eos_id

		state_path = os.path.join(run, '.state.yaml')
		for _ in range(2):
			config.save()
			state = yaml.safe_load(open(state_path, 'r'))
			assert state['_assets'] == {'UnifiedSeq2Vocab': 'unifiedSeq2Vocab.json'}
			assert state['data']['args']['vocab_path'] == 'unifiedSeq2Vocab.json'
			assert state['model']['args']['vocab_path'] == 'unifiedSeq2Vocab.json'
			# The distributed trainer reloads mid-epoch, so in memory the path must stay absolute.
			config.load()
			assert config['data.args.vocab_path'] == path
			assert config['model.args.vocab_path'] == path

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

		# Either direction terminates on the SAME shared <eos>; there is no modality-specific one.
		midi_target = mixed_state()
		midi_target['data']['args'] = {'source_format': 'lilylet', 'target_format': 'midiseq2'}
		midi_target_run = os.path.join(root, 'midi-target')
		os.makedirs(midi_target_run)
		midi_target_config = Configuration(midi_target_run, midi_target)
		assert midi_target_config['model.args.eos_id'] == tokenizer.eos_id

		wrong_eos = mixed_state()
		wrong_eos['model']['args']['eos_id'] = 264
		wrong_eos_run = os.path.join(root, 'wrong-eos')
		os.makedirs(wrong_eos_run)
		assert raises(ValueError, lambda: Configuration(wrong_eos_run, wrong_eos))

		# --- the generic assets: dispatch itself -------------------------------------------------
		# A config declaring no assets is untouched: no file, no _assets, no injection. This is the
		# midiseq2 -> midiseq2 case, which must stay exactly as it was.
		plain = mixed_state(assets=())
		plain.pop('assets')
		plain['data']['args'] = {'source_format': 'midiseq2', 'target_format': 'midiseq2'}
		plain_run = os.path.join(root, 'plain')
		os.makedirs(plain_run)
		plain_config = Configuration(plain_run, plain)
		assert os.listdir(plain_run) == ['.state.yaml']
		assert plain_config['model.args.vocab_size'] is None
		assert plain_config['model.args.vocab_path'] is None
		assert plain_config['_assets'] is None

		# An unregistered or malformed entry fails loudly rather than being skipped.
		for broken in ({'type': 'NoSuchAsset'}, {'args': {}}, 'UnifiedSeq2Vocab'):
			broken_run = os.path.join(root, f'broken-{abs(hash(str(broken)))}')
			os.makedirs(broken_run)
			assert raises(ValueError, lambda: Configuration(broken_run, mixed_state(assets=(broken,)))), broken
		not_a_list = mixed_state()
		not_a_list['assets'] = {'type': 'UnifiedSeq2Vocab'}
		list_run = os.path.join(root, 'assets-not-list')
		os.makedirs(list_run)
		assert raises(ValueError, lambda: Configuration(list_run, not_a_list))

		# Resume in a FRESH interpreter, where nothing has imported the builder's module yet: the
		# asset entry points must trigger the config's imports themselves. load() resolves references
		# before preprocess() runs, and the distributed trainer calls load() on its own mid-epoch.
		probe = ('import os, sys, yaml; sys.path.insert(0, %r)\n'
			'from starry.utils.registry import ASSETS\n'
			'assert not ASSETS, ASSETS\n'
			'from starry.utils.config import Configuration\n'
			'c = Configuration(%r)\n'
			'want = os.path.join(%r, "unifiedSeq2Vocab.json")\n'
			'assert c["model.args.vocab_path"] == want, c["model.args.vocab_path"]\n'
			'assert c["model.args.eos_id"] == 2\n'
			'c.load()\n'
			'assert c["model.args.vocab_path"] == want, "bare load() lost the absolute path"\n'
			'c.save()\n'
			'assert yaml.safe_load(open(os.path.join(%r, ".state.yaml")))["model"]["args"]'
			'["vocab_path"] == "unifiedSeq2Vocab.json"\n'
			'print("fresh-interpreter resume ok")') % (REPO_ROOT, run, run, run)
		result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
		assert result.returncode == 0, result.stderr
		assert 'fresh-interpreter resume ok' in result.stdout

		# Declared on resume but with no stored reference: the asset was never published.
		orphan = os.path.join(root, 'orphan')
		shutil.copytree(run, orphan)
		orphan_state = yaml.safe_load(open(os.path.join(orphan, '.state.yaml'), 'r'))
		del orphan_state['_assets']
		with open(os.path.join(orphan, '.state.yaml'), 'w') as f:
			yaml.dump(orphan_state, f)
		assert raises(ValueError, lambda: Configuration(orphan))

		# An absolute or nested reference would escape the run directory.
		for reference in ('/etc/passwd', 'sub/dir.json', ''):
			escape = os.path.join(root, f'escape-{abs(hash(reference))}')
			shutil.copytree(run, escape)
			escape_state = yaml.safe_load(open(os.path.join(escape, '.state.yaml'), 'r'))
			escape_state['_assets'] = {'UnifiedSeq2Vocab': reference}
			with open(os.path.join(escape, '.state.yaml'), 'w') as f:
				yaml.dump(escape_state, f)
			assert raises(ValueError, lambda: Configuration(escape)), reference

		# The v2 layout also held 1094 rows, so shape cannot distinguish it — the schema must.
		v2 = copy.deepcopy(artifact)
		v2['type'] = 'disjoint-lilylet-midiseq2'
		v2['version'] = 2
		v2_path = os.path.join(root, 'v2.json')
		with open(v2_path, 'w') as f:
			json.dump(v2, f)
		assert raises(ValueError, lambda: load_unified_vocab(v2_path))
		assert not UnifiedSeq2Tokenizer.matches(v2_path)
		stale = os.path.join(root, 'stale-run')
		shutil.copytree(run, stale)
		shutil.copy(v2_path, os.path.join(stale, 'unifiedSeq2Vocab.json'))
		assert raises(ValueError, lambda: Configuration(stale))

		truncated = copy.deepcopy(artifact)
		truncated['entries'] = truncated['entries'][:-1]
		truncated['vocab_size'] = len(truncated['entries'])
		truncated_path = os.path.join(root, 'truncated.json')
		with open(truncated_path, 'w') as f:
			json.dump(truncated, f)
		assert raises(ValueError, lambda: load_unified_vocab(truncated_path))

		# A relabelled control region must not pass as canonical.
		relabelled = copy.deepcopy(artifact)
		relabelled['entries'][6]['token'] = '<mask>'
		relabelled_path = os.path.join(root, 'relabelled.json')
		with open(relabelled_path, 'w') as f:
			json.dump(relabelled, f)
		assert raises(ValueError, lambda: load_unified_vocab(relabelled_path))

		unified_loss = MidiTranslatorLoss(vocab_path=path, d_model=16, n_layer=1, n_head=1,
			d_inner=32, max_seq_len=32, dropout=0)
		assert unified_loss.deducer.vocab_size == 1094
		types = unified_loss.type_of_id
		assert types.numel() == 1094
		assert int(types[tokenizer.sep_id]) == 7					# sep
		assert set(types[:16].tolist()) == {0, 7}					# controls: special + sep
		assert set(types[16:264].tolist()) == {8}					# Lilylet content -> err_lyl
		assert 8 not in set(types[264:].tolist())					# MIDI content keeps MIDI classes
		assert unified_loss.type_names[8] == 'err_lyl'
		assert unified_loss.ce_weight_of_id.numel() == 1094

		legacy_json = os.path.join(root, 'legacy.json')
		with open(legacy_json, 'w') as f:
			json.dump({'vocab': []}, f)
		assert not UnifiedSeq2Tokenizer.matches(legacy_json)
		legacy_loss = MidiTranslatorLoss(vocab_path=None, d_model=16, n_layer=1, n_head=1,
			d_inner=32, max_seq_len=16, dropout=0)
		assert legacy_loss.deducer.vocab_size == Midiseq2Tokenizer().vocab_size

		model = legacy_loss.deducer
		model.eos_id = 7
		steps = iter((torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]]),
			torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1.]])))
		seen_masks = []
		def scripted_forward (ids, masks=None, position_ids=None):
			seen_masks.append(None if masks is None else masks.clone())
			last = next(steps).to(ids.device)
			return last[:, None, :].expand(ids.shape[0], ids.shape[1], -1)
		model.forward = scripted_forward
		generated = model.generate(torch.tensor([1, 2]), max_new_tokens=2, masks=torch.ones(2, dtype=torch.long))
		assert generated.tolist() == [0, 7]
		assert [mask.shape[1] for mask in seen_masks] == [2, 3]
		assert seen_masks[1].tolist() == [[1, 1, 1]]

		# A volatile config still needs its assets to exist — the model reads vocab_path while being
		# built — but must not write them into the run directory it names. They go to a scratch dir
		# owned by the Configuration, so validation notebooks and the inference tools keep working.
		volatile = os.path.join(root, 'volatile')
		os.makedirs(volatile)
		volatile_config = Configuration(volatile, mixed_state(), volatile=True)
		assert os.listdir(volatile) == []
		scratch = volatile_config['model.args.vocab_path']
		assert os.path.dirname(scratch) == volatile_config.dir != volatile
		assert load_unified_vocab(scratch)['mapping_sha256'] == artifact['mapping_sha256']
		assert volatile_config['model.args.vocab_size'] == 1094
		# The scratch directory outlives creation: anything holding the config can still read the pin.
		del volatile_config
		assert not os.path.exists(scratch)

		roundtrip = os.path.join(root, 'roundtrip.json')
		write_unified_vocab(roundtrip, artifact)
		assert load_unified_vocab(roundtrip) == artifact

		midiseq2_asset(root)
	finally:
		shutil.rmtree(root)

	print('unified Seq2Seq2 lifecycle checks passed')


if __name__ == '__main__':
	main()
