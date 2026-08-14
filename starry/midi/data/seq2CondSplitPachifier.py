'''Conditioned-MIDI MEASUREWISE patchifier — midiseq2 SPLIT-MODALITY variant.

Sibling of starry.midi.data.seq2CondPachifier, differing ONLY in the on-disk layout: instead of
one joint [T, 64] tensor that left-aligns 16-wide lilylet patches into the midi frame (48-col <pad>
tail) and shares a single integer axis between two vocabularies, this packer keeps the two
modalities in SEPARATE tensors at their native widths and vocabularies:

    lyl_patches   int16 [Lp, lyl_patch_size]   lilylet ids (vocab 256)
    midi_patches  int16 [Mp, midi_patch_size]  midiseq2 ids (vocab 582)

Everything upstream is reused verbatim: patchify_lilylet (16-wide lyl patches + per-patch measure)
and patchify_midi_seq2 (64-wide midiseq2 measure-patches + own/src measure). The split packer simply
SKIPS the widen+concat that seq2CondPachifier.build_item does, and stores per-modality measure-id
arrays instead of the length-T joint modality/measures/src_measures.

Consumed by starry.midi.data.seq2CondSplitPatchy (Seq2CondSplitMidiPatchy).
'''

import os
from typing import Any, Dict, List, Tuple

import torch

from ...lilylet.data.patchifier import LilyletTokenizer
from .condPatchifier import patchify_lilylet, PATCH_SIZE as LYL_PATCH_SIZE
# reuse the midiseq2 tokenizer + measurewise midi patchify + constants verbatim.
from .seq2CondPachifier import (
	Midiseq2Tokenizer, patchify_midi_seq2, PATCH_SIZE,
	PAD_ID, BOS_ID, EOS_ID, UNKNOWN_ID, EOM_ID,
)


SPLIT_SHARDED_FORMAT = 'cond-midiseq2-measurewise-split-patches-sharded'


def build_item_split (sample: Dict[str, Any], lyl_text: str, midi_seq2_text: str,
	lyl_tokenizer: LilyletTokenizer, seq2_tokenizer: Midiseq2Tokenizer,
	patch_size: int = PATCH_SIZE, lyl_patch_size: int = LYL_PATCH_SIZE,
	patch_stream: bool = True) -> Dict[str, Any]:
	'''Build one SPLIT-MODALITY item: two separate patch tensors + per-modality measure ids.

	Calls the SAME two patchifiers as seq2CondPachifier.build_item but returns their outputs
	directly (no 16->64 widening, no concatenation). Lilylet patches stay at lyl_patch_size (16),
	midi patches at patch_size (64); no shared integer axis, no modality/lyl_count needed (each
	modality is its own tensor).

	Fields:
		lyl_patches  int16 [Lp, lyl_patch_size]   lilylet ids
		midi_patches int16 [Mp, patch_size]       midiseq2 ids
		lyl_meas     int16 [Lp]  own == src lilylet measure (0 = prompt/<bos>/header prefix)
		midi_meas    int16 [Mp]  own midi measure (0 = header)
		midi_src     int16 [Mp]  lilylet-aligned measure via source_measure (0 = header)
	'''
	lyl_patches, lyl_meas = patchify_lilylet(lyl_text, lyl_tokenizer, file=sample.get('id', ''),
		patch_size=lyl_patch_size, patch_stream=patch_stream)
	midi_patches, midi_mm, midi_src = patchify_midi_seq2(midi_seq2_text, sample['measures'],
		seq2_tokenizer, patch_size=patch_size)

	return dict(
		id=sample.get('id', ''),
		lyl_patches=torch.tensor(lyl_patches, dtype=torch.int16),
		midi_patches=torch.tensor(midi_patches, dtype=torch.int16),
		lyl_meas=torch.tensor(lyl_meas, dtype=torch.int16),
		midi_meas=torch.tensor(midi_mm, dtype=torch.int16),
		midi_src=torch.tensor(midi_src, dtype=torch.int16),
		lyl_patch_size=lyl_patch_size,
		midi_patch_size=patch_size,
		M_lyl=int(max(lyl_meas) if lyl_meas else 0),
		M_midi=len(sample['measures']),
	)


def pack_split (samples: List[Dict[str, Any]], lyl_root: str, midi_seq2_root: str,
	lyl_tokenizer: LilyletTokenizer, seq2_tokenizer: Midiseq2Tokenizer, out_path: str,
	patch_size: int = PATCH_SIZE, lyl_patch_size: int = LYL_PATCH_SIZE,
	patch_stream: bool = True) -> Dict[str, Any]:
	'''Pack samples into a single-shard SPLIT artifact (index .pt + one shard .pt beside it).

	Same on-disk layout / file-resolution as seq2CondPachifier.pack (lyl from
	<lyl_root>/<basename(sample['lyl'])>, midiseq2 from <midi_seq2_root>/<sample['id']>.midiseq2.txt),
	so seq2CondSplitPatchy._ItemStore loads it identically — only the format string and the stored
	per-item fields differ.
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
			items.append(build_item_split(sample, lyl_text, midi_seq2_text, lyl_tokenizer, seq2_tokenizer,
				patch_size=patch_size, lyl_patch_size=lyl_patch_size, patch_stream=patch_stream))
		except Exception as e:		# noqa: BLE001
			skipped.append((sample.get('id', ''), str(e)))

	torch.save(dict(items=items), os.path.join(out_dir, shard_name))
	index = dict(
		version=2,
		format=SPLIT_SHARDED_FORMAT,
		tokenizers=dict(
			lyl=dict(vocab_size=(max(lyl_tokenizer.id_by_token.values()) + 1) if lyl_tokenizer.id_by_token else 0),
			midi=dict(vocab_size=seq2_tokenizer.vocab_size),
		),
		config=dict(patch_size=patch_size, lyl_patch_size=lyl_patch_size,
			midi_patch_size=patch_size, patch_stream=patch_stream),
		shards=[dict(file=shard_name, count=len(items))],
		stats=dict(samples=len(samples), packed=len(items), skipped=len(skipped), skips=skipped),
	)
	torch.save(index, out_path)
	return index
