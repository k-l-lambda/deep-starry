'''Conditioned-MIDI measurewise patchifier.

Builds ONE joint sequence per song for the conditioned-MIDI generation task:

    [ lilylet patches (the score, measures 1..M_lyl) ]  ++  [ midi patches (events
    bucketed into measures 1..M_midi, each measure terminated by an <eom> patch) ]

The lilylet segment is a READ-ONLY condition (never a prediction target); the midi
segment is what the model learns to generate, coupled to the score through a custom
windowed/block attention mask built at batch time (see starry.midi.data.condPatchy).

Two modalities keep their OWN token ids and use SEPARATE patch-embedding modules in the
model, so this packer records, per patch, the modality (0=lilylet / 1=midi) and the
lilylet-segment length `lyl_count` (the split index). It also records, per patch, two
measure indices:
  - `measures`     : the patch's own modality-local measure (lyl j / midi i; 0 = prefix/header)
  - `src_measures` : the LILYLET-ALIGNED measure (lyl: own j; midi: source_measure(i) from
                     dataset.yaml; 0 = prefix/header). The cross mask aligns midi measure i to
                     lilylet measure src_measures, so a future repeat-EXPANDED midi (where i
                     and j differ) works unchanged.

This reuses starry.lilylet.data.patchifier (lilylet side) and starry.midi.tokenizer
(midi side) verbatim — only the per-measure tagging + concatenation + tick bucketing are
new here.
'''

import bisect
import os
from typing import Any, Dict, List, Tuple

import torch

from ...lilylet.data.patchifier import (
	LilyletTokenizer,
	split_lilylet_document, split_measures, split_voice_segments,
	split_patches, pad_patch, special_patch,
)
from ..tokenizer import MidiTokenizer, FIELD_EVENT_TOKENS, HEADER_EVENT_TOKENS


PATCH_SIZE = 16
# Artifact format string understood by CondMidiPatchy's _ItemStore.
SHARDED_FORMAT = 'cond-midi-measurewise-patches-sharded'

_FIELD_EVENTS = set(FIELD_EVENT_TOKENS)
_HEADER_EVENTS = set(HEADER_EVENT_TOKENS)


def patchify_lilylet (text: str, tokenizer: LilyletTokenizer, file: str = '',
	patch_size: int = PATCH_SIZE, patch_stream: bool = True) -> Tuple[List[List[int]], List[int]]:
	'''Lilylet score -> (patches, per-patch measure index).

	Mirrors starry.lilylet.data.patchifier.patchify_text, but: (1) emits NO trailing <eos>
	patch (the sequence continues into the midi segment), and (2) returns a parallel
	measure-index list (0 for the prompt/<bos>/header prefix, 1..M_lyl for body patches).
	With patch_stream the per-measure `[r:i/remaining]` position tag is preserved (each such
	tagged patch belongs to its measure j). Measures are sequential chunks, so the 1-based
	enumerate position equals the lilylet measure number.
	'''
	unknowns: Dict[Any, Any] = {}
	metadata_lines, body_lines = split_lilylet_document(text)
	measures = split_measures(body_lines)
	if patch_stream:
		total = len(measures)
		measures = [f'[r:{i}/{total - i - 1}]' + measure for i, measure in enumerate(measures)]

	patches: List[List[int]] = []
	measure_idx: List[int] = []

	# prefix: %-prompt patches, then <bos>, then [field] header patches; all measure 0.
	prompt_patches: List[List[int]] = []
	header_patches: List[List[int]] = []
	for line in metadata_lines:
		target = prompt_patches if line.lstrip().startswith('%') else header_patches
		target.extend(split_patches(tokenizer.encode(line, file, unknowns), patch_size, tokenizer.eos_id))
	for p in prompt_patches:
		patches.append(p); measure_idx.append(0)
	patches.append(special_patch('bos', patch_size, tokenizer.bos_id, tokenizer.eos_id))
	measure_idx.append(0)
	for p in header_patches:
		patches.append(p); measure_idx.append(0)

	# body: each measure j (1-based) -> voice-segment patches tagged with j.
	for j, chunk in enumerate(measures, start=1):
		for segment in split_voice_segments(chunk):
			for p in split_patches(tokenizer.encode(segment, file, unknowns), patch_size, tokenizer.eos_id):
				patches.append(p); measure_idx.append(j)

	patches = [pad_patch(p, patch_size, tokenizer.pad_id) for p in patches]
	return patches, measure_idx


def patchify_midi (text: str, measures_meta: List[Dict[str, Any]], tokenizer: MidiTokenizer,
	patch_size: int = PATCH_SIZE) -> Tuple[List[List[int]], List[int], List[int]]:
	'''Whole-song MidiText -> (patches, own midi-measure index, lilylet-aligned measure index).

	Walks the event lines accumulating absolute tick from each FIELD event's deltaTime, and
	buckets events into midi measures by the start_tick boundaries in `measures_meta` (the
	dataset.yaml `measures` list: [{index, start_tick, source_measure}, ...], 1-based index).

	Bucketing rule (the user's measure-boundary convention):
	  - note_on and all OTHER events: a tick exactly ON a boundary belongs to the LATER measure
	    (bisect_right).
	  - note_off: a tick on a boundary belongs to the EARLIER measure (bisect_right of t-1).
	Header events (ticks_per_beat / format_type / track) carry no deltaTime and are emitted
	as measure-0 patches before measure 1.

	An <eom> patch terminates each midi measure (inserted when the measure index advances, and
	once more after the last measure); the whole-song <eos> patch closes the sequence. No <bos>
	patch on the midi side (the lilylet segment already opened the joint sequence).

	Returns (patches, mm, src) where mm[k] is patch k's own midi measure (0 for header/eom-of-0
	never happens; eom carries the measure it terminates) and src[k] is the lilylet-aligned
	measure via source_measure (0 for header).
	'''
	boundaries = [m['start_tick'] for m in measures_meta]		# ascending, boundaries[0] == 0
	n_measures = len(measures_meta)
	# midi measure index (1-based) -> source (lilylet-aligned) measure
	src_of = {int(m['index']): int(m['source_measure']) for m in measures_meta if m.get('source_measure') is not None}

	def measure_of (tick: int, is_off: bool) -> int:
		t = tick - 1 if is_off else tick
		i = bisect.bisect_right(boundaries, t)		# 1-based measure (boundaries[0]=0 -> >=1)
		return min(max(i, 1), n_measures)

	eom_patch = pad_patch([tokenizer.eom_id], patch_size, tokenizer.pad_id)

	patches: List[List[int]] = []
	mm: List[int] = []		# own midi measure
	src: List[int] = []		# lilylet-aligned measure

	def emit (patch: List[int], own_measure: int):
		patches.append(patch)
		mm.append(own_measure)
		src.append(src_of.get(own_measure, 0) if own_measure > 0 else 0)

	# --- pass 1: parse events, accumulating each FIELD event's ABSOLUTE tick ---
	# Header events (no deltaTime) are emitted up front as measure-0 patches. FIELD events
	# are collected with their absolute tick + original content fields (after deltaTime), so
	# pass 2 can reorder and re-time them.
	header_lines: List[str] = []
	events = []		# list of (abst, is_off, head, rest_fields[list[str]])
	abst = 0
	for raw in text.split('\n'):
		line = raw.strip()
		if not line:
			continue
		head = line.split(' ', 1)[0]
		if head in _HEADER_EVENTS:
			header_lines.append(line)
			continue
		if head not in _FIELD_EVENTS:
			# excluded / unrecognized: still advance time so later ticks stay correct.
			parts = line.split(' ')
			if len(parts) > 1:
				try:
					abst += int(parts[1], 16)
				except ValueError:
					pass
			continue
		# FIELD event: deltaTime (parts[1]) is the gap BEFORE this event, so the absolute tick
		# is the running tick + this event's own delta.
		parts = line.split(' ')
		delta = int(parts[1], 16) if len(parts) > 1 else 0
		abst += delta
		events.append((abst, head == 'note_off', head, parts[2:]))

	# --- reorder so patch index -> measure index is STRICTLY monotonic ---
	# A note that sustains until a bar line ends exactly on the next measure's start tick; its
	# note_off carries the "ending belongs to the previous measure" rule. But in time order that
	# note_off is serialized AFTER the next measure's note_ons (same tick), which would make the
	# measure index dip by 1. Stable-sort each absolute-tick cluster with note_off FIRST: a
	# boundary note_off then precedes the next measure's onsets, so measure index never goes
	# backward. (Stable keeps original order within equal (tick, off-ness).)
	order = sorted(range(len(events)), key=lambda k: (events[k][0], 0 if events[k][1] else 1))
	events = [events[k] for k in order]

	# emit header patches first (measure 0).
	for line in header_lines:
		ids = tokenizer.encode_event(line)
		if ids is not None:
			emit(pad_patch(ids, patch_size, tokenizer.pad_id), 0)

	# --- pass 2: re-time (delta = gap from the previous event in the NEW order) + bucket ---
	cur_measure = 0			# measure of the patches emitted so far (0 = header region)
	prev_tick = 0
	for tick, is_off, head, rest in events:
		new_delta = tick - prev_tick
		prev_tick = tick
		m = measure_of(tick, is_off=is_off)
		# advancing into a new measure: close the measure(s) just left with <eom>.
		while cur_measure < m:
			if cur_measure >= 1:
				emit(eom_patch, cur_measure)
			cur_measure += 1
		line = ' '.join([head, format(new_delta, 'x')] + rest)
		ids = tokenizer.encode_event(line)
		if ids is not None:
			emit(pad_patch(ids, patch_size, tokenizer.pad_id), m)


	# close any remaining measures up to the last with <eom>.
	while cur_measure < n_measures:
		if cur_measure >= 1:
			emit(eom_patch, cur_measure)
		cur_measure += 1
	if n_measures >= 1:
		emit(eom_patch, n_measures)		# terminate the final measure

	# whole-song <eos> patch, attributed to the last measure.
	patches.append(special_patch('eos', patch_size, tokenizer.bos_id, tokenizer.eos_id))
	mm.append(n_measures)
	src.append(src_of.get(n_measures, 0))
	return patches, mm, src


def build_item (sample: Dict[str, Any], lyl_text: str, midi_text: str,
	lyl_tokenizer: LilyletTokenizer, midi_tokenizer: MidiTokenizer,
	patch_size: int = PATCH_SIZE, patch_stream: bool = True) -> Dict[str, Any]:
	'''Build one joint-sequence item dict from a dataset.yaml sample + its lyl/midi texts.

	Returns:
	  id          str
	  patches     uint8 [T, patch_size]   lilylet ids (0..255) then midi ids (0..40)
	  lyl_count   int                      number of lilylet patches L (the modality split)
	  measures    int16 [T]                per-patch own measure (lyl j / midi i; 0 prefix/header)
	  src_measures int16 [T]               per-patch lilylet-aligned measure
	  modality    uint8 [T]                0 = lilylet, 1 = midi
	  M_lyl, M_midi int
	'''
	lyl_patches, lyl_meas = patchify_lilylet(lyl_text, lyl_tokenizer, file=sample.get('id', ''),
		patch_size=patch_size, patch_stream=patch_stream)
	midi_patches, midi_mm, midi_src = patchify_midi(midi_text, sample['measures'], midi_tokenizer,
		patch_size=patch_size)

	L = len(lyl_patches)
	all_patches = lyl_patches + midi_patches
	# own measure: lilylet uses its body measure index; midi uses its own midi measure
	own_meas = lyl_meas + midi_mm
	# lilylet-aligned measure: lilylet patch -> its own measure (j); midi -> source_measure(i)
	src_meas = lyl_meas + midi_src
	modality = [0] * L + [1] * len(midi_patches)

	return dict(
		id=sample.get('id', ''),
		patches=torch.tensor(all_patches, dtype=torch.uint8),
		lyl_count=L,
		measures=torch.tensor(own_meas, dtype=torch.int16),
		src_measures=torch.tensor(src_meas, dtype=torch.int16),
		modality=torch.tensor(modality, dtype=torch.uint8),
		M_lyl=int(max(lyl_meas) if lyl_meas else 0),
		M_midi=len(sample['measures']),
	)


def pack (samples: List[Dict[str, Any]], lyl_root: str, midi_txt_root: str,
	lyl_tokenizer: LilyletTokenizer, midi_tokenizer: MidiTokenizer, out_path: str,
	patch_size: int = PATCH_SIZE, patch_stream: bool = True) -> Dict[str, Any]:
	'''Pack samples into a single-shard artifact at out_path (an index .pt + one shard .pt).

	lyl text is read from `<lyl_root>/<sample['lyl'] basename>` and midi whole-song text from
	`<midi_txt_root>/<sample['id']>.txt`. The shard file sits beside the index.
	'''
	out_dir = os.path.dirname(os.path.abspath(out_path))
	os.makedirs(out_dir, exist_ok=True)
	shard_name = os.path.splitext(os.path.basename(out_path))[0] + '.shard00000.pt'

	items: List[Dict[str, Any]] = []
	skipped: List[Tuple[str, str]] = []
	for sample in samples:
		lyl_path = os.path.join(lyl_root, os.path.basename(sample['lyl']))
		midi_path = os.path.join(midi_txt_root, sample['id'] + '.txt')
		if not (os.path.exists(lyl_path) and os.path.exists(midi_path)):
			skipped.append((sample.get('id', ''), 'missing lyl or midi-txt'))
			continue
		with open(lyl_path, encoding='utf-8') as f:
			lyl_text = f.read()
		with open(midi_path, encoding='utf-8') as f:
			midi_text = f.read()
		try:
			items.append(build_item(sample, lyl_text, midi_text, lyl_tokenizer, midi_tokenizer,
				patch_size=patch_size, patch_stream=patch_stream))
		except Exception as e:		# noqa: BLE001
			skipped.append((sample.get('id', ''), str(e)))

	torch.save(dict(items=items), os.path.join(out_dir, shard_name))
	index = dict(
		version=2,
		format=SHARDED_FORMAT,
		tokenizers=dict(
			lyl=dict(vocab_size=lyl_tokenizer.id_by_token and max(lyl_tokenizer.id_by_token.values()) + 1),
			midi=dict(vocab_size=midi_tokenizer.vocab_size),
		),
		config=dict(patch_size=patch_size, patch_stream=patch_stream),
		shards=[dict(file=shard_name, count=len(items))],
		stats=dict(samples=len(samples), packed=len(items), skipped=len(skipped), skips=skipped),
	)
	torch.save(index, out_path)
	return index


