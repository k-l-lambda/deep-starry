'''Merged-control vocabulary for mixed Seq2Seq2 runs.

The authoritative inputs remain the Lilylet and midiseq2 assets. A mixed training run builds this
mapping once and pins the resulting artifact in its Configuration directory; resumed runs load that
exact file rather than rebuilding against whatever assets happen to be in the current checkout.

Layout (v3). One canonical control region is shared by both modalities, and only the CONTENT half of
each source vocabulary is appended:

	0..15		canonical controls + reserve — <pad> <bos> <eos> <unknown> <mask> <sep> <eom>, then
				<reserved_7>..<reserved_15>. Both modalities use these same ids, so a mixed sample has
				exactly one embedding per control rather than one per (control, modality) pair.
	16..263		Lilylet source-local ids 8..255
	264..1093	midiseq2 source-local ids 8..837

Equal content STRINGS across modalities (digits, 'a'..'f', '-', '_') stay distinct ids: the blocks are
disjoint and there is deliberately no global string lookup, only modality-scoped ones.

Neither source tokenizer is modified. `LilyletTokenizer` emits ASCII byte VALUES as local ids, so its
output has to be remapped through `lilylet_id` here; that is why Lilylet content cannot simply keep
its source ids.

This is an id-semantic break from the v2 `disjoint-lilylet-midiseq2` layout, which appended both
complete vocabularies (and therefore duplicated the controls). Both layouts happen to hold 1094 rows,
so shape alone cannot distinguish them — `validate_unified_vocab` rejects v2 on type and version, and
a v2 mixed checkpoint cannot be resumed under this mapping.
'''

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from ...utils.assets import VocabAsset, publish_atomically
from ...utils.registry import register_asset
from .seq2CondPachifier import _ASSET_VOCAB as MIDI_ASSET, _load_vocab


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LILYLET_ASSET = os.path.join(ROOT, 'assets', 'lilylet-tokenizer.json')
ARTIFACT_TYPE = 'merged-lilylet-midiseq2'
ARTIFACT_VERSION = 3

# The shared control region. Sized to 16 so the reserve absorbs future controls without moving a
# single content id — the whole point of pinning the artifact per run.
SPECIAL_SIZE = 16
SPECIAL_TOKENS: List[str] = ['<pad>', '<bos>', '<eos>', '<unknown>', '<mask>', '<sep>', '<eom>'] \
	+ [f'<reserved_{i}>' for i in range(7, SPECIAL_SIZE)]
SPECIAL_IDS: Dict[str, int] = {
	'pad': 0, 'bos': 1, 'eos': 2, 'unknown': 3, 'mask': 4, 'sep': 5, 'eom': 6}

# Where each source vocabulary's CONTENT starts. Both assets carry 8 controls at locals 0..7.
CONTENT_LOCAL_START = 8
# Source control prefixes, asserted before they are dropped. They agree except at local 4.
LILYLET_SPECIAL_PREFIX: List[str] = ['<pad>', '<bos>', '<eos>', '<unknown>', '<mask>', '<sep>',
	'<reserved_6>', '<reserved_7>']
MIDI_SPECIAL_PREFIX: List[str] = ['<pad>', '<bos>', '<eos>', '<unknown>', '<eom>', '<sep>',
	'<reserved_6>', '<reserved_7>']
# Source control local id -> shared id. The source reserves are deliberately absent: nothing emits
# them, and mapping them would make two source ids collide on one shared reserve slot.
_LILYLET_SPECIAL_MAP: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
_MIDI_SPECIAL_MAP: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 5}


def _digest_json (value: Any) -> str:
	payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8')
	return hashlib.sha256(payload).hexdigest()


def _is_int (value: Any) -> bool:
	return isinstance(value, int) and not isinstance(value, bool)


def _load_lilylet_vocab (path: str) -> List[str]:
	with open(path, 'r', encoding='utf-8') as f:
		artifact = json.load(f)
	vocab = sorted(artifact['vocab'], key=lambda entry: entry['id'])
	if [entry['id'] for entry in vocab] != list(range(len(vocab))):
		raise ValueError(f'{path}: Lilylet vocabulary ids must be contiguous from 0')
	return [entry['token'] for entry in vocab]


def build_unified_vocab (lilylet_path: str = LILYLET_ASSET,
	midiseq2_path: str = MIDI_ASSET) -> Dict[str, Any]:
	'''Build the canonical merged-control artifact from the two authoritative vocabularies.'''
	lyl_tokens = _load_lilylet_vocab(lilylet_path)
	midi_tokens = _load_vocab(midiseq2_path)
	# Assert what is about to be dropped. A source asset that renumbered its controls would otherwise
	# shift every content id silently.
	if lyl_tokens[:CONTENT_LOCAL_START] != LILYLET_SPECIAL_PREFIX:
		raise ValueError(f'{lilylet_path}: unexpected Lilylet control prefix '
			f'{lyl_tokens[:CONTENT_LOCAL_START]}')
	if midi_tokens[:CONTENT_LOCAL_START] != MIDI_SPECIAL_PREFIX:
		raise ValueError(f'{midiseq2_path}: unexpected midiseq2 control prefix '
			f'{midi_tokens[:CONTENT_LOCAL_START]}')

	lyl_content = lyl_tokens[CONTENT_LOCAL_START:]
	midi_content = midi_tokens[CONTENT_LOCAL_START:]
	lyl_offset = SPECIAL_SIZE
	midi_offset = lyl_offset + len(lyl_content)

	entries: List[Dict[str, Any]] = [
		dict(id=i, modality='special', local_id=i, token=token)
		for i, token in enumerate(SPECIAL_TOKENS)]
	entries.extend(dict(id=lyl_offset + i, modality='lilylet', local_id=CONTENT_LOCAL_START + i,
		token=token) for i, token in enumerate(lyl_content))
	entries.extend(dict(id=midi_offset + i, modality='midiseq2', local_id=CONTENT_LOCAL_START + i,
		token=token) for i, token in enumerate(midi_content))

	mapping = [(entry['modality'], entry['local_id'], entry['token']) for entry in entries]
	artifact = {
		'version': ARTIFACT_VERSION,
		'type': ARTIFACT_TYPE,
		'vocab_size': len(entries),
		'blocks': {
			'special': {'offset': 0, 'size': SPECIAL_SIZE, 'local_start': 0},
			'lilylet': {'offset': lyl_offset, 'size': len(lyl_content),
				'local_start': CONTENT_LOCAL_START},
			'midiseq2': {'offset': midi_offset, 'size': len(midi_content),
				'local_start': CONTENT_LOCAL_START},
		},
		'special_ids': dict(SPECIAL_IDS),
		'provenance': {
			'lilylet_content_sha256': _digest_json(lyl_content),
			'midiseq2_content_sha256': _digest_json(midi_content),
		},
		'mapping_sha256': _digest_json(mapping),
		'entries': entries,
	}
	return validate_unified_vocab(artifact)


def _validate_blocks (blocks: Any) -> Dict[str, Dict[str, int]]:
	if not isinstance(blocks, dict) or set(blocks) != {'special', 'lilylet', 'midiseq2'}:
		raise ValueError('unified vocabulary requires exactly special/lilylet/midiseq2 blocks')
	for name in ('special', 'lilylet', 'midiseq2'):
		block = blocks[name]
		if not isinstance(block, dict) or set(block) != {'offset', 'size', 'local_start'} \
			or not all(_is_int(block[key]) and block[key] >= 0 for key in block):
			raise ValueError(f'unified vocabulary {name} block geometry is invalid')
	if blocks['special'] != {'offset': 0, 'size': SPECIAL_SIZE, 'local_start': 0}:
		raise ValueError('unified vocabulary control block must be 16 ids at offset 0')
	if blocks['lilylet']['offset'] != SPECIAL_SIZE \
		or blocks['midiseq2']['offset'] != SPECIAL_SIZE + blocks['lilylet']['size']:
		raise ValueError('unified vocabulary blocks must be contiguous: special, Lilylet, midiseq2')
	if blocks['lilylet']['local_start'] != CONTENT_LOCAL_START \
		or blocks['midiseq2']['local_start'] != CONTENT_LOCAL_START:
		raise ValueError(f'unified vocabulary content blocks must start at source local id '
			f'{CONTENT_LOCAL_START}')
	return blocks


def validate_unified_vocab (artifact: Dict[str, Any]) -> Dict[str, Any]:
	'''Validate a serialized mapping without consulting today's source assets.'''
	if not isinstance(artifact, dict):
		raise ValueError('unified vocabulary artifact must be a JSON object')
	if artifact.get('type') != ARTIFACT_TYPE:
		raise ValueError(f'not a {ARTIFACT_TYPE!r} vocabulary artifact')
	if artifact.get('version') != ARTIFACT_VERSION:
		raise ValueError(f'unsupported unified vocabulary version {artifact.get("version")!r}')
	blocks = _validate_blocks(artifact.get('blocks'))
	entries = artifact.get('entries')
	if not isinstance(entries, list):
		raise ValueError('unified vocabulary requires entries')
	total = blocks['midiseq2']['offset'] + blocks['midiseq2']['size']
	if not _is_int(artifact.get('vocab_size')):
		raise ValueError('unified vocabulary size must be an integer')
	if artifact['vocab_size'] != len(entries) or len(entries) != total:
		raise ValueError('unified vocabulary size does not match its blocks/entries')

	for name in ('special', 'lilylet', 'midiseq2'):
		block = blocks[name]
		for i in range(block['size']):
			entry = entries[block['offset'] + i]
			if not isinstance(entry, dict) or entry.get('id') != block['offset'] + i \
				or entry.get('modality') != name \
				or entry.get('local_id') != block['local_start'] + i \
				or not isinstance(entry.get('token'), str):
				raise ValueError(f'invalid unified vocabulary entry at id {block["offset"] + i}')

	if [entry['token'] for entry in entries[:SPECIAL_SIZE]] != SPECIAL_TOKENS:
		raise ValueError('unified vocabulary control tokens do not match the canonical layout')
	if artifact.get('special_ids') != SPECIAL_IDS:
		raise ValueError('unified vocabulary special ids do not match the canonical layout')

	mapping = [(entry['modality'], entry['local_id'], entry['token']) for entry in entries]
	if artifact.get('mapping_sha256') != _digest_json(mapping):
		raise ValueError('unified vocabulary mapping digest mismatch')

	provenance = artifact.get('provenance')
	if not isinstance(provenance, dict):
		raise ValueError('unified vocabulary provenance is missing')
	blockwise = {name: [entry['token'] for entry in
			entries[blocks[name]['offset']:blocks[name]['offset'] + blocks[name]['size']]]
		for name in ('lilylet', 'midiseq2')}
	for name, tokens in blockwise.items():
		# Uniqueness is per block only: identical strings across modalities are distinct tokens here.
		if len(set(tokens)) != len(tokens):
			raise ValueError(f'unified vocabulary {name} tokens must be unique within their block')
		if provenance.get(f'{name}_content_sha256') != _digest_json(tokens):
			raise ValueError('unified vocabulary provenance digest mismatch')
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

	def write (f):
		json.dump(artifact, f, ensure_ascii=False, indent=2)
		f.write('\n')

	publish_atomically(path, write)


class UnifiedSeq2Tokenizer:
	'''Tokenizer metadata over an in-memory or run-local unified artifact.

	Controls are MODALITY-NEUTRAL: `bos_id`, `eos_id`, `sep_id`, `pad_id`, `unknown_id`, `mask_id` and
	`eom_id` are the ids to use on either arm. There is deliberately no `midiseq2_bos_id` and no global
	token->id lookup; content is reached through the modality-scoped `lilylet_id` / `midi_id` remaps.
	'''

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
		self.mask_id = self.special_ids['mask']
		self.sep_id = self.special_ids['sep']
		self.eom_id = self.special_ids['eom']
		self.lilylet_offset = self.blocks['lilylet']['offset']
		self.midiseq2_offset = self.blocks['midiseq2']['offset']
		# Local -> unified tables. -1 marks a source id with no unified counterpart (the source
		# reserves), so a stray one raises instead of aliasing a live control.
		self._lilylet_map = self._local_map('lilylet', _LILYLET_SPECIAL_MAP)
		self._midi_map = self._local_map('midiseq2', _MIDI_SPECIAL_MAP)
		self.midiseq2_id_by_token = {
			self.tokens[self.midiseq2_offset + i]: self.midiseq2_offset + i
			for i in range(self.blocks['midiseq2']['size'])}
		self.lilylet_id_by_token = {
			self.tokens[self.lilylet_offset + i]: self.lilylet_offset + i
			for i in range(self.blocks['lilylet']['size'])}

	def _local_map (self, modality: str, specials: Dict[int, int]) -> List[int]:
		block = self.blocks[modality]
		table = [-1] * (CONTENT_LOCAL_START + block['size'])
		for local_id, unified in specials.items():
			table[local_id] = unified
		for i in range(block['size']):
			table[CONTENT_LOCAL_START + i] = block['offset'] + i
		return table

	def _remap (self, table: Sequence[int], modality: str, local_id: int) -> int:
		if not 0 <= local_id < len(table) or table[local_id] < 0:
			raise ValueError(f'{modality} local id has no unified mapping: {local_id}')
		return table[local_id]

	def midi_id (self, local_id: int) -> int:
		'''midiseq2 source-local id -> unified id. Local 4 (<eom>) becomes the shared <eom>.'''
		return self._remap(self._midi_map, 'midiseq2', local_id)

	def lilylet_id (self, local_id: int) -> int:
		'''Lilylet source-local id -> unified id. Locals 8..255 are content, NOT unified ids.'''
		return self._remap(self._lilylet_map, 'lilylet', local_id)

	def lilylet_ids (self, local_ids: Sequence[int]) -> List[int]:
		'''Remap a whole `LilyletTokenizer.encode()` result.'''
		table = self._lilylet_map
		return [self._remap(table, 'lilylet', local_id) for local_id in local_ids]


@register_asset
class UnifiedSeq2Vocab (VocabAsset):
	'''Run-local unified vocabulary, declared by a mixed config's `assets:` list.

	A mixed Lilylet/midiseq2 run needs this mapping to exist before its feeder and model are built,
	and needs the SAME mapping back on resume — rebuilding from whatever assets are in the checkout
	would silently reinterpret the checkpoint's embedding rows. Unlike its sibling `Midiseq2Vocab`,
	which copies an authoritative file, this one SYNTHESIZES the mapping from two of them, so there is
	no single asset a run could be pointed back at.
	'''

	FILENAME = 'unifiedSeq2Vocab.json'

	@staticmethod
	def publish (path, args):
		write_unified_vocab(path, build_unified_vocab(**args) if args else build_unified_vocab())

	@staticmethod
	def describe (path):
		artifact = load_unified_vocab(path)
		# One shared <eos> terminates the generated half in EITHER mixed direction.
		return {'vocab_size': artifact['vocab_size'], 'eos_id': artifact['special_ids']['eos']}
