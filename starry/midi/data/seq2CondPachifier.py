'''Conditioned-MIDI MEASUREWISE patchifier — midiseq2 variant.

Sibling of starry.midi.data.condPatchifier, but the MIDI side is the `midiseq2` self-describing
token language (assets/midiseq2Vocab.yaml, vocab 838) instead of the one-event-per-patch
MidiTokenizer, and patches are cut on MEASURE boundaries rather than on event boundaries.

TWO-STAGE pipeline (per the design): basic whole-song MidiText is first converted to midiseq2
TEXT by the authoritative TS grammar (intelli-piano/tools/midiTxtToSeq2.ts -> a `midi-seq2/`
dir of `.midiseq2.txt`); THIS module only CONSUMES that midiseq2 text (Midiseq2Tokenizer parses
it — there is no Python renderer). Parsing midiseq2 text means accumulating the leading `E…`
elapse run of each event into an absolute tick, so events can be bucketed into measures by the
dataset.yaml start_tick boundaries.

Builds ONE joint sequence per song:

    [ lilylet patches (the score, measures 1..M_lyl) ]  ++  [ midiseq2 patches (each MEASURE's
    token run, chunked into patch_size-wide patches, terminated by an <eom> TOKEN) ]

Key differences from condPatchifier:
  - MIDI tokens are midiseq2 ids (0..837), so the joint patch tensor is int16 (not uint8).
  - patch_size defaults to 64 (a measure averages ~102 midiseq2 tokens; see the fmenu stat).
  - A measure is one variable-length midiseq2 token RUN, ending with an <eom> token (id 4),
    then right-padded to a whole number of patch_size-wide patches. Patch boundaries ALWAYS
    align to measure boundaries: no patch straddles two measures (a short measure pads to one
    patch; a long measure spans several, its tail padded). There is NO standalone <eom> patch.
  - The midiseq2 elapse convention puts delta tokens BEFORE their event; a measure is re-timed
    self-contained (the first event's elapse run = offset from the bar start tick).

Per-patch bookkeeping mirrors condPatchifier so starry.midi.data.seq2CondPatchy /
starry.midi.data.condPatchy build_vis can couple midi measures to lilylet measures unchanged:
  - `measures`     : the patch's own modality-local measure (lyl j / midi i; 0 = prefix/header)
  - `src_measures` : the lilylet-aligned measure (lyl own j; midi source_measure(i); 0 = header)

The lilylet side reuses starry.lilylet.data.patchifier verbatim (read-only condition, must
match the frozen encoder's training distribution), including its patch_size — see build_item.
'''

import bisect
import os
from typing import Any, Dict, List, Tuple

import torch

from ...lilylet.data.patchifier import LilyletTokenizer
from ...utils.assets import VocabAsset, publish_atomically
from ...utils.registry import register_asset
# The lilylet side is identical to condPatchifier's, so reuse its patchify_lilylet verbatim
# (emits no trailing <eos> patch, returns per-patch measure indices). NOTE: the lilylet
# patch_size must match the frozen encoder's training patch_size (16), which is INDEPENDENT
# of the midiseq2 patch_size (64) used for the midi side — see build_item.
from .condPatchifier import patchify_lilylet, PATCH_SIZE as LYL_PATCH_SIZE


PATCH_SIZE = 64
SHARDED_FORMAT = 'cond-midiseq2-measurewise-patches-sharded'

# midiseq2 special ids (from assets/midiseq2Vocab.yaml, section "Special control tokens").
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNKNOWN_ID = 3
EOM_ID = 4
# <sep> occupies what was <reserved_5>, so adding it shifted no content id and vocab_size is
# unchanged (838). Used by starry.midi.data.seq2seq2 to join a source and target sequence.
SEP_ID = 5

_ASSET_VOCAB = os.path.join(
	os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
	'assets', 'midiseq2Vocab.yaml')


# ---------------------------------------------------------------------------------------
# midiseq2 tokenizer — Python port of intelli-piano/inc/midiseq2 (encoder.ts + grammar).
# Validated token-for-token against the authoritative TS `textToMidiseq2` grammar.
# ---------------------------------------------------------------------------------------

# event field layouts, after the deltaTime (mirrors FIELD_EVENT_SPECS / the jison grammar).
_HEADER_EVENTS = {'ticks_per_beat', 'format_type'}
_FIELD_EVENTS = {
	'set_tempo', 'time_signature', 'key_signature', 'sequence_number', 'channel_prefix',
	'smpte_offset', 'end_of_track', 'note_on', 'note_off', 'polytouch', 'control_change',
	'program_change', 'aftertouch', 'pitchwheel',
}


def _load_vocab (path: str) -> List[str]:
	'''Parse assets/midiseq2Vocab.yaml (a comment-annotated JSON-ish token list) -> tokens.

	The file is a bracketed, comma-separated list with `#` comments and optionally-quoted
	entries (quotes protect `#..`, `$..`, `_`, `-`, digit strings from YAML). We strip
	comments/brackets and unquote, preserving order == id.
	'''
	tokens: List[str] = []
	with open(path, 'r', encoding='utf-8') as f:
		for raw in f:
			s = raw.strip()
			if not s or s in ('[', ']'):
				continue
			if s.startswith('#'):
				continue
			s = s.rstrip(',').strip()
			if not s or s.startswith('#'):
				continue
			if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
				s = s[1:-1]
			tokens.append(s)
	return tokens


class Midiseq2Tokenizer:
	'''Vocab + parser for midiseq2 text (the TS-grammar output).

	Vocab comes from assets/midiseq2Vocab.yaml (order == id). This is the CONSUMER side of the
	two-stage pipeline: `parse_events` reads midiseq2 text into absolute-tick events, `token_ids`
	maps token strings -> ids (<unknown> = id 3 for anything off-vocab), and `_elapse` regenerates
	a measure's leading delta run after re-timing. Basic-MidiText -> midiseq2 RENDERING is NOT
	done here — the TS grammar (intelli-piano/tools/midiTxtToSeq2.ts) owns that.
	'''

	def __init__ (self, vocab_path: str = _ASSET_VOCAB):
		self.tokens = _load_vocab(vocab_path)
		self.id_by_token = {t: i for i, t in enumerate(self.tokens)}
		self.vocab_size = len(self.tokens)
		self.pad_id, self.bos_id, self.eos_id = PAD_ID, BOS_ID, EOS_ID
		self.unknown_id, self.eom_id, self.sep_id = UNKNOWN_ID, EOM_ID, SEP_ID
		# sanity: the special block must be where the module constants say it is. Checked over the
		# whole named block, not just its ends — every packed artifact and trained checkpoint reads
		# these ids positionally, so a reordered vocab file has to fail here rather than silently
		# retrain against shifted ids.
		expected = {'<pad>': PAD_ID, '<bos>': BOS_ID, '<eos>': EOS_ID, '<unknown>': UNKNOWN_ID,
			'<eom>': EOM_ID, '<sep>': SEP_ID}
		actual = {name: self.id_by_token.get(name) for name in expected}
		assert actual == expected, f'midiseq2 vocab special-token layout mismatch: {actual} != {expected}'

	# --- elapse (delta) run encoding — the only rendering the Python side still does: it
	# regenerates each measure's leading E… run after re-timing from the bar start. The rest of
	# midiseq2 rendering (basic MidiText -> midiseq2) is done by the TS grammar upstream
	# (intelli-piano/tools/midiTxtToSeq2.ts); this module only PARSES midiseq2 text.

	@staticmethod
	def _elapse (delta: int) -> List[str]:
		'''Delta (ticks) -> canonical midiseq2 elapse-token run (mirrors the TS grammar's
		`elapse`): one E1000 per 0x1000, then the mid nibble-byte, then the low nibble.'''
		ticks = delta
		out: List[str] = []
		while ticks >= 0x1000:
			out.append('E1000'); ticks -= 0x1000
		mid = ticks & 0xff0
		if mid:
			out.append('E' + format(mid, 'x').zfill(3))
		low = ticks & 0xf
		if low:
			out.append('E' + format(low, 'x'))
		return out

	def token_ids (self, token_strings: List[str]) -> List[int]:
		'''Map rendered token strings -> ids (<unknown> for anything off-vocab).'''
		return [self.id_by_token.get(t, self.unknown_id) for t in token_strings]

	# --- midiseq2 TEXT parsing (consume the TS-grammar output; mirrors parser.ts) ---

	@staticmethod
	def _is_elapse (tok: str) -> bool:
		if len(tok) < 2 or tok[0] != 'E':
			return False
		try:
			int(tok[1:], 16)
			return True
		except ValueError:
			return False

	def parse_events (self, text: str) -> Tuple[List[str], List[Tuple[int, bool, List[str]]]]:
		'''Parse midiseq2 text -> (header_token_strings, events).

		Accumulates each event's leading `E…` elapse run into an absolute tick. Returns the
		header token run (ticks_per_beat / format_type, no time) and a list of
		(abs_tick, is_note_off, body_token_strings) where body excludes the elapse tokens
		(they are regenerated per measure at emit time). The event keyword stays in body[0].

		Whitespace (space or newline) is the only separator, matching the TS grammar's `\\s+`
		lexer — so we tokenize the whole document flat and re-group by keyword.
		'''
		toks = text.split()
		header: List[str] = []
		events: List[Tuple[int, bool, List[str]]] = []
		abst = 0
		i = 0
		n = len(toks)
		# leading header keywords (ticks_per_beat / format_type) have no elapse and come first.
		while i < n and toks[i] in _HEADER_EVENTS:
			head = toks[i]; i += 1
			body = [head]
			while i < n and toks[i] not in _HEADER_EVENTS and toks[i] not in _FIELD_EVENTS \
					and not self._is_elapse(toks[i]):
				body.append(toks[i]); i += 1
			header.append(body)
		# flatten header runs into one token list (each is [keyword, ...nibbles]).
		header_tokens = [t for run in header for t in run]

		# body events: optional elapse run, then a keyword + its args (until the next elapse or
		# keyword). Elapse accumulates onto the running absolute tick.
		while i < n:
			tok = toks[i]
			if self._is_elapse(tok):
				abst += int(tok[1:], 16); i += 1
				continue
			if tok in _HEADER_EVENTS or tok in _FIELD_EVENTS:
				head = tok; i += 1
				body = [head]
				while i < n and toks[i] not in _HEADER_EVENTS and toks[i] not in _FIELD_EVENTS \
						and not self._is_elapse(toks[i]):
					body.append(toks[i]); i += 1
				events.append((abst, head == 'note_off', body))
			else:
				# stray token (shouldn't happen on grammar output); skip defensively.
				i += 1
		return header_tokens, events


@register_asset
class Midiseq2Vocab (VocabAsset):
	'''Run-local COPY of assets/midiseq2Vocab.yaml, declared by a config's `assets:` list.

	The vocabulary is positional — every checkpoint's embedding rows and every packed artifact read
	these ids by index — so a run that reads the repository asset at load time is one asset edit away
	from silently reinterpreting its own weights. Pinning a copy beside the checkpoint makes the
	mapping part of the run rather than part of the checkout, and lets inference recover the exact
	vocabulary a checkpoint was trained against (see tools/midi/translateMidiseq2.py).

	Nothing is transformed: the file is copied verbatim, so `Midiseq2Tokenizer(vocab_path)` reads the
	pinned copy exactly as it reads the asset. Contrast `UnifiedSeq2Vocab`, which synthesizes a mixed
	mapping from two assets and therefore has no single file to copy.
	'''

	FILENAME = 'midiseq2Vocab.yaml'

	@staticmethod
	def publish (path, args):
		source = args.get('source') or _ASSET_VOCAB
		with open(source, 'r', encoding='utf-8') as f:
			text = f.read()
		# Parse before publishing, so an unreadable or reordered vocabulary fails while creating the
		# run rather than on its first resume.
		Midiseq2Tokenizer(source)
		publish_atomically(path, lambda f: f.write(text))

	@staticmethod
	def describe (path):
		# Constructing the tokenizer asserts the special-token block sits where the module constants
		# say it does, so a reordered or truncated pin cannot reach the model.
		return {'vocab_size': Midiseq2Tokenizer(path).vocab_size, 'eos_id': EOS_ID}


# ---------------------------------------------------------------------------------------
# measurewise midi patchify (midiseq2)
# ---------------------------------------------------------------------------------------

def _chunk_measure (ids: List[int], patch_size: int) -> List[List[int]]:
	'''Split one measure's token-id run into patch_size-wide patches, right-padded with <pad>.

	Patch boundaries align to the measure: a run of length <= patch_size becomes one padded
	patch; a longer run becomes ceil(len/patch_size) patches, the last one padded. No patch
	ever mixes two measures (the caller passes exactly one measure's run, <eom> included).
	'''
	if not ids:
		return []
	patches: List[List[int]] = []
	for start in range(0, len(ids), patch_size):
		chunk = ids[start:start + patch_size]
		if len(chunk) < patch_size:
			chunk = chunk + [PAD_ID] * (patch_size - len(chunk))
		patches.append(chunk)
	return patches


def patchify_midi_seq2 (seq2_text: str, measures_meta: List[Dict[str, Any]], tokenizer: Midiseq2Tokenizer,
	patch_size: int = PATCH_SIZE) -> Tuple[List[List[int]], List[int], List[int]]:
	'''midiseq2 TEXT -> (midiseq2 patches, own midi-measure index, lilylet-aligned index).

	`seq2_text` is the output of the TS grammar (midiTxtToSeq2.ts). Parses it into events with
	absolute ticks (accumulating each event's leading E… elapse run), buckets events into midi
	measures by the dataset.yaml start_tick boundaries (identical rule to
	condPatchifier.patchify_midi: note_off on a boundary -> earlier measure; every other event
	-> later measure), RE-TIMES each measure self-contained by regenerating the leading elapse
	run from the bar-start tick, appends an <eom> TOKEN, and chunks the run into patch_size-wide
	patches. The header token run (ticks_per_beat/format_type) becomes the measure-0 patch(es).
	The whole-song <eos> patch closes the sequence.

	Returns (patches, mm, src): patches int lists length patch_size; mm[k] the patch's own midi
	measure; src[k] the lilylet-aligned measure via source_measure (0 for header).
	'''
	boundaries = [m['start_tick'] for m in measures_meta]
	n_measures = len(measures_meta)
	assert n_measures >= 1, 'empty measures_meta'
	assert boundaries[0] == 0, f'first measure start_tick must be 0, got {boundaries[0]}'
	assert all(boundaries[i] <= boundaries[i + 1] for i in range(n_measures - 1)), \
		'measure start_ticks must be non-decreasing'
	indices = [int(m['index']) for m in measures_meta]
	assert indices == list(range(1, n_measures + 1)), \
		f'measure indices must be 1..{n_measures}, got {indices[:8]}...'
	for m in measures_meta:
		sm = m.get('source_measure')
		assert sm is not None and int(sm) >= 1, \
			f'measure {m.get("index")} has invalid source_measure {sm!r} (needed for cross-attention)'
	src_of = {int(m['index']): int(m['source_measure']) for m in measures_meta}

	def measure_of (tick: int, is_off: bool) -> int:
		t = tick - 1 if is_off else tick
		i = bisect.bisect_right(boundaries, t)
		return min(max(i, 1), n_measures)

	# --- parse the midiseq2 text into header tokens + absolute-tick events ---
	header_tokens, events = tokenizer.parse_events(seq2_text)

	# stable-sort each abstick cluster with note_off FIRST so patch->measure stays monotonic
	# (a boundary note_off precedes the next measure's onsets). Same rule as condPatchifier.
	order = sorted(range(len(events)), key=lambda k: (events[k][0], 0 if events[k][1] else 1))
	events = [events[k] for k in order]

	patches: List[List[int]] = []
	mm: List[int] = []
	src: List[int] = []

	# --- measure-0 header patch: the parsed header token run (no elapse) ---
	if header_tokens:
		for patch in _chunk_measure(tokenizer.token_ids(header_tokens), patch_size):
			patches.append(patch); mm.append(0); src.append(0)

	# --- bucket events into measures; regenerate each measure's elapse from the bar start ---
	cur_measure = 0
	buf_tokens: List[str] = []
	prev_tick = 0

	def flush_measure (measure: int):
		'''Emit the buffered token run for `measure` (+ <eom>) as measure-aligned patches.'''
		if measure < 1:
			buf_tokens.clear()
			return
		run = list(buf_tokens) + ['<eom>']
		ids = tokenizer.token_ids(run)
		for patch in _chunk_measure(ids, patch_size):
			patches.append(patch); mm.append(measure); src.append(src_of.get(measure, 0))
		buf_tokens.clear()

	for tick, is_off, body in events:
		m = measure_of(tick, is_off=is_off)
		# advancing measures: flush the measure(s) just left (empty ones emit an <eom>-only run).
		while cur_measure < m:
			if cur_measure >= 1:
				flush_measure(cur_measure)
			cur_measure += 1
			prev_tick = boundaries[cur_measure - 1]		# reset delta origin to the new bar start
		delta = tick - prev_tick
		prev_tick = tick
		# regenerate the leading elapse run (self-contained within the measure) + the event body.
		buf_tokens += tokenizer._elapse(delta) + body

	# flush the final in-progress measure, then close any remaining empty measures.
	if cur_measure >= 1:
		flush_measure(cur_measure)
	while cur_measure < n_measures:
		cur_measure += 1
		for patch in _chunk_measure(tokenizer.token_ids(['<eom>']), patch_size):
			patches.append(patch); mm.append(cur_measure); src.append(src_of.get(cur_measure, 0))

	# whole-song <eos> patch (bos-led, matching the lilylet special_patch convention).
	eos_patch = [BOS_ID] + [EOS_ID] * (patch_size - 1)
	patches.append(eos_patch); mm.append(n_measures); src.append(src_of.get(n_measures, 0))
	return patches, mm, src


def build_item (sample: Dict[str, Any], lyl_text: str, midi_seq2_text: str,
	lyl_tokenizer: LilyletTokenizer, seq2_tokenizer: Midiseq2Tokenizer,
	patch_size: int = PATCH_SIZE, lyl_patch_size: int = LYL_PATCH_SIZE,
	patch_stream: bool = True) -> Dict[str, Any]:
	'''Build one joint-sequence item from a dataset.yaml sample + its lyl text and midiseq2 text.

	`midi_seq2_text` is the TS-grammar midiseq2 output (a `midi-seq2/` file), NOT basic MidiText.

	The joint patch tensor is [T, patch_size] int16. Lilylet patches are produced at the
	frozen-encoder width `lyl_patch_size` (16) and LEFT-ALIGNED into the wider (64) frame,
	right-padded with <pad> — the model slices columns [0, lyl_patch_size) for the encoder.
	Midi patches fill the full width. Modality (0=lyl / 1=midi) disambiguates the two.

	Returns dict with the CondMidiPatchy contract fields plus lyl_patch_size (so the feeder /
	model can recover the encoder width from the artifact).
	'''
	lyl_patches, lyl_meas = patchify_lilylet(lyl_text, lyl_tokenizer, file=sample.get('id', ''),
		patch_size=lyl_patch_size, patch_stream=patch_stream)
	midi_patches, midi_mm, midi_src = patchify_midi_seq2(midi_seq2_text, sample['measures'],
		seq2_tokenizer, patch_size=patch_size)

	assert patch_size >= lyl_patch_size, \
		f'patch_size ({patch_size}) must be >= lyl_patch_size ({lyl_patch_size})'
	# widen lilylet patches into the joint frame (left-aligned, <pad>-filled tail).
	pad_tail = patch_size - lyl_patch_size
	lyl_wide = [p + [lyl_tokenizer.pad_id] * pad_tail for p in lyl_patches]

	L = len(lyl_wide)
	all_patches = lyl_wide + midi_patches
	own_meas = lyl_meas + midi_mm
	src_meas = lyl_meas + midi_src
	modality = [0] * L + [1] * len(midi_patches)

	return dict(
		id=sample.get('id', ''),
		patches=torch.tensor(all_patches, dtype=torch.int16),
		lyl_count=L,
		measures=torch.tensor(own_meas, dtype=torch.int16),
		src_measures=torch.tensor(src_meas, dtype=torch.int16),
		modality=torch.tensor(modality, dtype=torch.uint8),
		lyl_patch_size=lyl_patch_size,
		M_lyl=int(max(lyl_meas) if lyl_meas else 0),
		M_midi=len(sample['measures']),
	)


def pack (samples: List[Dict[str, Any]], lyl_root: str, midi_seq2_root: str,
	lyl_tokenizer: LilyletTokenizer, seq2_tokenizer: Midiseq2Tokenizer, out_path: str,
	patch_size: int = PATCH_SIZE, lyl_patch_size: int = LYL_PATCH_SIZE,
	patch_stream: bool = True) -> Dict[str, Any]:
	'''Pack samples into a single-shard artifact (index .pt + one shard .pt beside it).

	lyl text is read from <lyl_root>/<basename(sample['lyl'])>, midiseq2 text from
	<midi_seq2_root>/<sample['id']>.midiseq2.txt (the TS-grammar output). Mirrors
	condPatchifier.pack (same on-disk layout) so the seq2CondPatchy _ItemStore loads it the same.
	'''
	out_dir = os.path.dirname(os.path.abspath(out_path))
	os.makedirs(out_dir, exist_ok=True)
	shard_name = os.path.splitext(os.path.basename(out_path))[0] + '.shard00000.pt'

	items: List[Dict[str, Any]] = []
	skipped: List[Tuple[str, str]] = []
	for sample in samples:
		lyl_path = os.path.join(lyl_root, os.path.basename(sample['lyl']))
		midi_path = os.path.join(midi_seq2_root, sample['id'] + '.midiseq2.txt')
		if not (os.path.exists(lyl_path) and os.path.exists(midi_path)):
			skipped.append((sample.get('id', ''), 'missing lyl or midi-seq2'))
			continue
		with open(lyl_path, encoding='utf-8') as f:
			lyl_text = f.read()
		with open(midi_path, encoding='utf-8') as f:
			midi_seq2_text = f.read()
		try:
			items.append(build_item(sample, lyl_text, midi_seq2_text, lyl_tokenizer, seq2_tokenizer,
				patch_size=patch_size, lyl_patch_size=lyl_patch_size, patch_stream=patch_stream))
		except Exception as e:		# noqa: BLE001
			skipped.append((sample.get('id', ''), str(e)))

	torch.save(dict(items=items), os.path.join(out_dir, shard_name))
	index = dict(
		version=2,
		format=SHARDED_FORMAT,
		tokenizers=dict(
			lyl=dict(vocab_size=(max(lyl_tokenizer.id_by_token.values()) + 1) if lyl_tokenizer.id_by_token else 0),
			midi=dict(vocab_size=seq2_tokenizer.vocab_size),
		),
		config=dict(patch_size=patch_size, lyl_patch_size=lyl_patch_size, patch_stream=patch_stream),
		shards=[dict(file=shard_name, count=len(items))],
		stats=dict(samples=len(samples), packed=len(items), skipped=len(skipped), skips=skipped),
	)
	torch.save(index, out_path)
	return index
