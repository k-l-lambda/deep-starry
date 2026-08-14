'''Lilylet-first disjoint vocabulary for mixed Seq2Seq2 runs.

The authoritative inputs remain the Lilylet and midiseq2 assets. A mixed training run builds this
mapping once and pins the resulting artifact in its Configuration directory; resumed runs load that
exact file rather than rebuilding against whatever assets happen to be in the current checkout.
'''

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from .seq2CondPachifier import _ASSET_VOCAB as MIDI_ASSET, _load_vocab


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LILYLET_ASSET = os.path.join(ROOT, 'assets', 'lilylet-tokenizer.json')
ARTIFACT_TYPE = 'disjoint-lilylet-midiseq2'
ARTIFACT_VERSION = 2


def _digest_json (value: Any) -> str:
	payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8')
	return hashlib.sha256(payload).hexdigest()


def build_unified_vocab (lilylet_path: str = LILYLET_ASSET,
	midiseq2_path: str = MIDI_ASSET) -> Dict[str, Any]:
	'''Build the canonical Lilylet-first artifact from the two authoritative vocabularies.'''
	with open(lilylet_path, 'r', encoding='utf-8') as f:
		lilylet = json.load(f)
	lyl_vocab = sorted(lilylet['vocab'], key=lambda entry: entry['id'])
	if [entry['id'] for entry in lyl_vocab] != list(range(len(lyl_vocab))):
		raise ValueError(f'{lilylet_path}: Lilylet vocabulary ids must be contiguous from 0')
	midi_vocab = _load_vocab(midiseq2_path)
	offset = len(lyl_vocab)
	entries = [dict(id=entry['id'], modality='lilylet', local_id=entry['id'], token=entry['token'])
		for entry in lyl_vocab]
	entries.extend(dict(id=offset + i, modality='midiseq2', local_id=i, token=token)
		for i, token in enumerate(midi_vocab))
	mapping = [(entry['modality'], entry['local_id'], entry['token']) for entry in entries]
	artifact = {
		'version': ARTIFACT_VERSION,
		'type': ARTIFACT_TYPE,
		'vocab_size': len(entries),
		'blocks': {
			'lilylet': {'offset': 0, 'size': len(lyl_vocab)},
			'midiseq2': {'offset': offset, 'size': len(midi_vocab)},
		},
		'special_ids': {
			'pad': 0, 'bos': 1, 'eos': 2, 'unknown': 3, 'sep': 5,
			'lilylet_mask': 4,
			'midiseq2_pad': offset,
			'midiseq2_bos': offset + 1,
			'midiseq2_eos': offset + 2,
			'midiseq2_unknown': offset + 3,
			'midiseq2_eom': offset + 4,
			'midiseq2_sep': offset + 5,
		},
		'provenance': {
			'lilylet_tokens_sha256': _digest_json([entry['token'] for entry in lyl_vocab]),
			'midiseq2_tokens_sha256': _digest_json(midi_vocab),
		},
		'mapping_sha256': _digest_json(mapping),
		'entries': entries,
	}
	return validate_unified_vocab(artifact)


def validate_unified_vocab (artifact: Dict[str, Any]) -> Dict[str, Any]:
	'''Validate a serialized mapping without consulting today's source assets.'''
	if not isinstance(artifact, dict):
		raise ValueError('unified vocabulary artifact must be a JSON object')
	if artifact.get('type') != ARTIFACT_TYPE:
		raise ValueError(f'not a {ARTIFACT_TYPE!r} vocabulary artifact')
	if artifact.get('version') != ARTIFACT_VERSION:
		raise ValueError(f'unsupported unified vocabulary version {artifact.get("version")!r}')
	blocks = artifact.get('blocks')
	entries = artifact.get('entries')
	if not isinstance(blocks, dict) or not isinstance(entries, list):
		raise ValueError('unified vocabulary requires blocks and entries')
	lyl = blocks.get('lilylet')
	midi = blocks.get('midiseq2')
	if not isinstance(lyl, dict) or not isinstance(midi, dict):
		raise ValueError('unified vocabulary requires lilylet and midiseq2 blocks')
	if not isinstance(lyl.get('offset'), int) or isinstance(lyl.get('offset'), bool) \
		or not isinstance(lyl.get('size'), int) or isinstance(lyl.get('size'), bool) \
		or not isinstance(midi.get('offset'), int) or isinstance(midi.get('offset'), bool) \
		or not isinstance(midi.get('size'), int) or isinstance(midi.get('size'), bool) \
		or lyl['offset'] < 0 or lyl['size'] < 0 or midi['offset'] < 0 or midi['size'] < 0:
		raise ValueError('unified vocabulary block offsets and sizes must be non-negative integers')
	if not isinstance(artifact.get('vocab_size'), int) or isinstance(artifact.get('vocab_size'), bool):
		raise ValueError('unified vocabulary size must be an integer')
	if lyl.get('offset') != 0 or midi.get('offset') != lyl.get('size'):
		raise ValueError('unified vocabulary blocks must be contiguous and Lilylet-first')
	if artifact.get('vocab_size') != len(entries) or len(entries) != midi.get('offset') + midi.get('size'):
		raise ValueError('unified vocabulary size does not match its blocks/entries')
	for i, entry in enumerate(entries):
		modality = 'lilylet' if i < midi['offset'] else 'midiseq2'
		local_id = i if modality == 'lilylet' else i - midi['offset']
		if not isinstance(entry, dict) or entry.get('id') != i or entry.get('modality') != modality \
			or entry.get('local_id') != local_id or not isinstance(entry.get('token'), str):
			raise ValueError(f'invalid unified vocabulary entry at id {i}')
	mapping = [(entry['modality'], entry['local_id'], entry['token']) for entry in entries]
	if artifact.get('mapping_sha256') != _digest_json(mapping):
		raise ValueError('unified vocabulary mapping digest mismatch')
	provenance = artifact.get('provenance')
	if not isinstance(provenance, dict):
		raise ValueError('unified vocabulary provenance is missing')
	lyl_tokens = [entry['token'] for entry in entries[:midi['offset']]]
	midi_tokens = [entry['token'] for entry in entries[midi['offset']:]]
	if len(set(lyl_tokens)) != len(lyl_tokens) or len(set(midi_tokens)) != len(midi_tokens):
		raise ValueError('unified vocabulary tokens must be unique within each modality block')
	if provenance.get('lilylet_tokens_sha256') != _digest_json(lyl_tokens) \
		or provenance.get('midiseq2_tokens_sha256') != _digest_json(midi_tokens):
		raise ValueError('unified vocabulary provenance digest mismatch')
	special = artifact.get('special_ids')
	expected = {
		'pad': 0, 'bos': 1, 'eos': 2, 'unknown': 3, 'sep': 5, 'lilylet_mask': 4,
		'midiseq2_pad': midi['offset'], 'midiseq2_bos': midi['offset'] + 1,
		'midiseq2_eos': midi['offset'] + 2, 'midiseq2_unknown': midi['offset'] + 3,
		'midiseq2_eom': midi['offset'] + 4, 'midiseq2_sep': midi['offset'] + 5,
	}
	if special != expected:
		raise ValueError('unified vocabulary special ids do not match the canonical layout')
	expected_tokens = {
		0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unknown>', 4: '<mask>', 5: '<sep>',
		midi['offset']: '<pad>', midi['offset'] + 1: '<bos>', midi['offset'] + 2: '<eos>',
		midi['offset'] + 3: '<unknown>', midi['offset'] + 4: '<eom>', midi['offset'] + 5: '<sep>',
	}
	if any(entries[i]['token'] != token for i, token in expected_tokens.items()):
		raise ValueError('unified vocabulary special token strings are invalid')
	return artifact


def load_unified_vocab (path: str) -> Dict[str, Any]:
	try:
		with open(path, 'r', encoding='utf-8') as f:
			return validate_unified_vocab(json.load(f))
	except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
		raise ValueError(f'invalid unified vocabulary {path!r}: {exc}') from exc


def write_unified_vocab (path: str, artifact: Optional[Dict[str, Any]] = None) -> None:
	'''Atomically publish a validated artifact.'''
	artifact = validate_unified_vocab(artifact if artifact is not None else build_unified_vocab())
	directory = os.path.dirname(os.path.abspath(path))
	fd, temporary = tempfile.mkstemp(prefix='.unified-vocab-', suffix='.tmp', dir=directory)
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as f:
			json.dump(artifact, f, ensure_ascii=False, indent=2)
			f.write('\n')
			f.flush()
			os.fsync(f.fileno())
		os.replace(temporary, path)
	finally:
		if os.path.exists(temporary):
			os.remove(temporary)


class UnifiedSeq2Tokenizer:
	'''Tokenizer metadata over an in-memory or run-local unified artifact.'''

	@staticmethod
	def matches (vocab_path: str) -> bool:
		try:
			with open(vocab_path, 'r', encoding='utf-8') as f:
				return json.load(f).get('type') == ARTIFACT_TYPE
		except (OSError, ValueError, AttributeError):
			return False

	def __init__ (self, vocab_path: Optional[str] = None, artifact: Optional[Dict[str, Any]] = None):
		if vocab_path is not None and artifact is not None:
			raise ValueError('provide vocab_path or artifact, not both')
		self.artifact = load_unified_vocab(vocab_path) if vocab_path else validate_unified_vocab(
			artifact if artifact is not None else build_unified_vocab())
		self.vocab_size = int(self.artifact['vocab_size'])
		self.blocks = self.artifact['blocks']
		self.special_ids = self.artifact['special_ids']
		self.tokens: List[str] = [entry['token'] for entry in self.artifact['entries']]
		self.pad_id = self.special_ids['pad']
		self.bos_id = self.special_ids['bos']
		self.eos_id = self.special_ids['eos']
		self.unknown_id = self.special_ids['unknown']
		self.sep_id = self.special_ids['sep']
		self.lilylet_mask_id = self.special_ids['lilylet_mask']
		self.midiseq2_offset = self.blocks['midiseq2']['offset']
		self.midiseq2_pad_id = self.special_ids['midiseq2_pad']
		self.midiseq2_bos_id = self.special_ids['midiseq2_bos']
		self.midiseq2_eos_id = self.special_ids['midiseq2_eos']
		self.midiseq2_unknown_id = self.special_ids['midiseq2_unknown']
		self.midiseq2_eom_id = self.special_ids['midiseq2_eom']
		self.midiseq2_sep_id = self.special_ids['midiseq2_sep']
		self.midiseq2_id_by_token = {
			self.tokens[self.midiseq2_offset + i]: self.midiseq2_offset + i
			for i in range(self.blocks['midiseq2']['size'])
		}

	def midi_id (self, local_id: int) -> int:
		if not 0 <= local_id < self.blocks['midiseq2']['size']:
			raise ValueError(f'midiseq2 local id out of range: {local_id}')
		return self.midiseq2_offset + local_id

	def lilylet_id (self, local_id: int) -> int:
		if not 0 <= local_id < self.blocks['lilylet']['size']:
			raise ValueError(f'lilylet local id out of range: {local_id}')
		return local_id
