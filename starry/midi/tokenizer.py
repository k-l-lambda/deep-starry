'''MidiText tokenizer (SkyTNT-style event patches).

Tokenizes the line-per-event MIDI text produced by music-widgets' MidiText codec
(`source/inc/MidiText.ts`). Following SkyTNT/midi-model, ONE EVENT = ONE PATCH:
each text line becomes a fixed-length patch whose first token is the reserved
event-name token and whose remaining tokens are the character-level serialization
of the rest of the line (hex digits + spaces), padded with <pad> to `patch_size`.

Vocabulary
----------
- Special tokens : <pad> <bos> <eos> <unknown>            (ids 0..3)
- Event tokens   : the reserved event-name tokens, one per MidiText token name
                   (note_on / note_off / control_change / set_tempo / ... plus the
                   header tokens ticks_per_beat / format_type / track). These are
                   matched as WHOLE tokens (the first whitespace-delimited field).
- Content tokens : the characters '0'-'9', 'a'-'f', ' ' (space), and '-' (sign).
                   The rest-of-line of every retained event is pure hex+space, so
                   this set is complete (see EXCLUDED below).

Excluded event types
--------------------
TEXT_SPECS (text/copyright/track_name/instrument_name/lyrics/marker/cue_point) and
DATA_SPECS (sysex/divided_sysex/sequencer_specific/meta_unknown) carry free text or
raw-byte payloads — the ONLY source of non-hex characters in the corpus — so their
lines are dropped entirely (event + payload), exactly as requested. What remains is
purely structured numeric events whose content is `[0-9a-f -]` + space.

This mirrors the in-repo `LilyletTokenizer` conventions (JSON-free here; the vocab
is built deterministically in code) and feeds a two-level patch/token model the same
way `starry/lilylet/data/patchifier.py` does.
'''

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# --- event-name tokens, mirroring MidiText.ts FIELD_SPECS (numeric events only) ---
# Kept in sync with music-widgets source/inc/MidiText.ts. Each entry maps the MidiText
# token to its ordered numeric fields (AFTER the deltaTime, which every field event
# carries as its implicit first value). Header tokens are handled separately below
# (they carry a single value and NO deltaTime).
FIELD_EVENT_SPECS: Dict[str, List[str]] = {
	'note_off':        ['channel', 'noteNumber', 'velocity'],
	'note_on':         ['channel', 'noteNumber', 'velocity'],
	'polytouch':       ['channel', 'noteNumber', 'amount'],
	'control_change':  ['channel', 'controllerType', 'value'],
	'program_change':  ['channel', 'programNumber'],
	'aftertouch':      ['channel', 'amount'],
	'pitchwheel':      ['channel', 'value'],
	'set_tempo':       ['microsecondsPerBeat'],
	'key_signature':   ['key', 'scale'],
	'sequence_number': ['number'],
	'channel_prefix':  ['channel'],
	'end_of_track':    [],
}
FIELD_EVENT_TOKENS: List[str] = list(FIELD_EVENT_SPECS.keys())

# Header lines carry a SINGLE value and no deltaTime: `ticks_per_beat 1e0`, `track 0`.
HEADER_EVENT_SPECS: Dict[str, List[str]] = {
	'ticks_per_beat': ['ticksPerBeat'],
	'format_type':    ['formatType'],
	'track':          ['trackIndex'],
}
HEADER_EVENT_TOKENS: List[str] = list(HEADER_EVENT_SPECS.keys())

# Maximum hexadecimal width (number of hex digits) each numeric field can occupy,
# derived from the byte width music-widgets' parser (MIDI/midifile.ts) reads it at —
# so the theoretical maximum event length is COMPUTED, not guessed or scanned.
#   - deltaTime         : MIDI varint, up to 4 bytes = 28 bits (max 0xFFFFFFF, 7 hex);
#                         capped here at 4 hex (0xFFFF = 65535 ticks ≈ tens of beats at
#                         typical tpb) as a practical bound — a longer gap (very rare) is
#                         truncated by pad_patch, never silently corrupted
#   - microsecondsPerBeat: setTempo, 3 bytes = 24 bits        -> 6 hex
#   - value             : controller value is 7-bit, but pitchBend reuses `value` at
#                         14 bits (0x3FFF) -> 4 hex (take the wider of the shared name)
#   - number / ticksPerBeat: int16                            -> 4 hex
#   - single-byte 0..255 fields (note/velocity/controller/...) -> 2 hex
#   - channel (4-bit)   : 1 hex
FIELD_MAX_HEX: Dict[str, int] = {
	'deltaTime': 4,
	'channel': 1,
	'noteNumber': 2, 'velocity': 2, 'amount': 2,
	'controllerType': 2, 'value': 4, 'programNumber': 2,
	'microsecondsPerBeat': 6,
	'key': 2, 'scale': 1,
	'number': 4,
	# header values
	'ticksPerBeat': 4, 'formatType': 1, 'trackIndex': 2,
}


def _event_max_tokens(fields: List[str], with_delta: bool) -> int:
	'''Theoretical max token count of one event line = 1 event token + the char-level
	serialization of its values. Values are space-separated hex; with_delta prepends the
	implicit deltaTime value. tokens = 1(event) + sum(hex widths) + (n_values - 1)(spaces).'''
	widths = ([FIELD_MAX_HEX['deltaTime']] if with_delta else []) + [FIELD_MAX_HEX[f] for f in fields]
	if not widths:
		return 1
	return 1 + sum(widths) + (len(widths) - 1)


def theoretical_max_event_tokens() -> int:
	'''The largest possible event-patch length across all retained event types — exact,
	from field byte-widths (no corpus scan). Used as the default patch_size so every
	well-formed event fits without truncation.'''
	field_max = max((_event_max_tokens(f, with_delta=True) for f in FIELD_EVENT_SPECS.values()), default=1)
	header_max = max((_event_max_tokens(f, with_delta=False) for f in HEADER_EVENT_SPECS.values()), default=1)
	return max(field_max, header_max)


# Default patch width: theoretical max retained-event length (15) rounded up to 16.
DEFAULT_PATCH_SIZE = 16


# Excluded: free-text / raw-byte events. Lines whose first token is one of these are
# dropped wholesale (they are the only non-hex content in MidiText output).
EXCLUDED_EVENT_TOKENS = frozenset([
	# TEXT_SPECS
	'text', 'copyright', 'track_name', 'instrument_name', 'lyrics', 'marker', 'cue_point',
	# DATA_SPECS
	'sysex', 'divided_sysex', 'sequencer_specific', 'meta_unknown',
	# rare/unused structured events excluded so they do not inflate patch_size:
	#  - smpte_offset: 6 fields, only a leading SMPTE marker
	#  - time_signature: notation-only — does NOT affect audio (sound timing is set by
	#    set_tempo + ticks_per_beat; the meter is a score/metronome concern)
	'smpte_offset',
	'time_signature',
])

# content characters (rest-of-line, after the event token)
CONTENT_CHARS: List[str] = list('0123456789abcdef') + [' ', '-']

SPECIAL_TOKENS: List[str] = ['<pad>', '<bos>', '<eos>', '<unknown>']


@dataclass
class UnknownHit:
	char: str
	count: int = 0


class MidiTokenizer:
	'''SkyTNT-style event-patch tokenizer for MidiText output.

	Vocab layout (ids are stable / contiguous):
	  0..3            special : <pad> <bos> <eos> <unknown>
	  4..             event   : FIELD_EVENT_TOKENS + HEADER_EVENT_TOKENS (whole-token)
	  ...             content : '0'-'9' 'a'-'f' ' ' '-'  (char-level)
	'''

	def __init__(self, patch_size: Optional[int] = None):
		# Default patch_size = DEFAULT_PATCH_SIZE (16): the theoretical max retained-event
		# length is 15 (control_change, computed from field byte-widths), rounded up to 16
		# for a little headroom. Every well-formed retained event fits without truncation.
		# Pass an explicit int to override (smaller = may truncate; larger = extra padding).
		self.patch_size = patch_size if patch_size is not None else DEFAULT_PATCH_SIZE

		tokens: List[str] = list(SPECIAL_TOKENS)
		self.event_tokens: List[str] = list(FIELD_EVENT_TOKENS) + list(HEADER_EVENT_TOKENS)
		tokens += self.event_tokens
		tokens += CONTENT_CHARS

		self.id_by_token: Dict[str, int] = {tok: i for i, tok in enumerate(tokens)}
		self.token_by_id: Dict[int, str] = {i: tok for tok, i in self.id_by_token.items()}
		self.vocab_size = len(tokens)

		self.pad_id = self.id_by_token['<pad>']
		self.bos_id = self.id_by_token['<bos>']
		self.eos_id = self.id_by_token['<eos>']
		self.unknown_id = self.id_by_token['<unknown>']

		self.event_id_set = {self.id_by_token[t] for t in self.event_tokens}
		self.excluded = set(EXCLUDED_EVENT_TOKENS)

	# --- single-line (one event) encode / decode ----------------------------

	def encode_event(self, line: str, unknowns: Optional[Dict[str, UnknownHit]] = None) -> Optional[List[int]]:
		'''Encode ONE MidiText line into a token-id list (event token + content chars).

		Returns None when the line is blank or its event type is EXCLUDED (text/data) —
		the caller drops it. Unknown content chars map to <unknown> (and are recorded in
		`unknowns` if provided). The returned list is NOT padded to patch_size.
		'''
		line = line.rstrip('\n')
		if not line.strip():
			return None

		sp = line.find(' ')
		head = line if sp < 0 else line[:sp]
		rest = '' if sp < 0 else line[sp + 1:]

		if head in self.excluded:
			return None
		event_id = self.id_by_token.get(head)
		if event_id is None:
			# unrecognized event name — treat the whole line as unknown event (dropped)
			return None

		ids: List[int] = [event_id]
		for ch in rest:
			tid = self.id_by_token.get(ch)
			if tid is None:
				ids.append(self.unknown_id)
				if unknowns is not None:
					hit = unknowns.get(ch)
					if hit is None:
						unknowns[ch] = hit = UnknownHit(char=ch)
					hit.count += 1
			else:
				ids.append(tid)
		return ids

	def decode_event(self, ids: List[int]) -> str:
		'''Inverse of encode_event: token-ids -> one MidiText line (pad/bos/eos stripped).'''
		out: List[str] = []
		for i, tid in enumerate(ids):
			if tid in (self.pad_id, self.bos_id, self.eos_id):
				continue
			tok = self.token_by_id.get(tid, '')
			if i == 0 and tid in self.event_id_set:
				out.append(tok + ' ')          # event name, then a space before content
			elif tid == self.unknown_id:
				out.append('')                 # unknown content char — unrecoverable
			else:
				out.append(tok)
		return ''.join(out).rstrip()

	# --- patch-level (one event = one patch) --------------------------------

	def pad_patch(self, ids: List[int]) -> List[int]:
		'''Pad/truncate one event's token-ids to exactly patch_size.

		Mirror lilylet.data.patchifier.split_patches(): when a patch is not already full,
		insert <eos> before right-padding with <pad>. For MIDI, one event = one patch, so
		this in-patch <eos> is the supervised event-boundary marker; without it the
		token-level LM never learns where the event's argument list ends.
		'''
		if len(ids) >= self.patch_size:
			return ids[:self.patch_size]
		ids = ids + [self.eos_id]
		return ids + [self.pad_id] * (self.patch_size - len(ids))

	def special_patch(self, kind: str) -> List[int]:
		'''A whole-patch boundary marker, matching patchifier.py's convention:
		bos = [<bos> ... <bos> <eos>], eos = [<bos> <eos> ... <eos>].'''
		if kind == 'bos':
			return [self.bos_id] * (self.patch_size - 1) + [self.eos_id]
		if kind == 'eos':
			return [self.bos_id] + [self.eos_id] * (self.patch_size - 1)
		raise ValueError(f'Unknown special patch kind: {kind}')

	def encode_patches(
		self,
		text: str,
		add_special_patches: bool = True,
		max_patches: Optional[int] = None,
		unknowns: Optional[Dict[str, UnknownHit]] = None,
	) -> Tuple[List[List[int]], List[str]]:
		'''Encode a full MidiText document into a list of fixed-size event patches.

		Each non-excluded line -> one patch (event token + content chars, padded to
		patch_size). Optional leading <bos> and trailing <eos> special patches frame the
		sequence. Returns (patches, dropped_event_names) where dropped lists the event
		names skipped (excluded or unrecognized) for transparency.
		'''
		patches: List[List[int]] = []
		dropped: List[str] = []

		if add_special_patches:
			patches.append(self.special_patch('bos'))

		for raw in text.split('\n'):
			line = raw.rstrip('\n')
			if not line.strip():
				continue
			ids = self.encode_event(line, unknowns=unknowns)
			if ids is None:
				head = line.split(' ', 1)[0]
				dropped.append(head)
				continue
			patches.append(self.pad_patch(ids))

		if add_special_patches:
			patches.append(self.special_patch('eos'))

		if max_patches is not None and len(patches) > max_patches:
			patches = patches[:max_patches]

		return patches, dropped

	def decode_patches(self, patches: List[List[int]]) -> str:
		'''Inverse of encode_patches: drop boundary-only patches, decode each event line.'''
		lines: List[str] = []
		for patch in patches:
			# skip pure boundary patches (all bos/eos/pad)
			body = [t for t in patch if t not in (self.pad_id, self.bos_id, self.eos_id)]
			if not body:
				continue
			line = self.decode_event(patch)
			if line:
				lines.append(line)
		return '\n'.join(lines)
