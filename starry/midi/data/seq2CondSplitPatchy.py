'''Seq2CondSplitMidiPatchy — feeder for the SPLIT-MODALITY conditioned-MIDI midiseq2 dataset.

Sibling of starry.midi.data.seq2CondPatchy (Seq2CondMidiPatchy), consuming the artifact written by
starry.midi.data.seq2CondSplitPachifier.pack_split. Same measurewise midiseq2 idea and the same
measure-coupling windows, but the two modalities are kept in SEPARATE tensors at their native widths
and vocabularies rather than one joint [T, 64] frame:

    lyl_patches  [Lp, lyl_patch_size=16]  lilylet ids (vocab 256)
    midi_patches [Mp, midi_patch_size=64] midiseq2 ids (vocab 838)

Batch contract (consumed by MidiSeq2PrefixInTokenLevel):
	lyl_patches    LongTensor [B, Lmax, lyl_patch_size]   lilylet ids
	lyl_masks      LongTensor [B, Lmax]                   1 = real lyl patch
	lyl_meas       LongTensor [B, Lmax]                   lyl measure (own == src; 0 = prefix/header)
	midi_patches   LongTensor [B, Mmax, midi_patch_size]  midiseq2 ids
	midi_masks     LongTensor [B, Mmax]                   1 = real midi patch
	midi_meas      LongTensor [B, Mmax]                   own midi measure (0 = header)
	midi_src       LongTensor [B, Mmax]                   lilylet-aligned measure (0 = header)
	midi_targets   LongTensor [B, Mmax]                   1 = supervised next-patch target
	midi_positions LongTensor [B, Mmax]                   0..Mp-1 (midi RoPE frame)
	midi_attn_mask BoolTensor [B, 1, Mmax, Mmax]          midi->midi windowed mask (True = attend)

The midi->lyl CROSS coupling is NOT emitted as a mask here: MidiSeq2PrefixInTokenLevel derives the
per-patch lyl cross window directly from midi_src / lyl_meas / w_cross. Only the midi self-attention
window mask (midi->midi block of build_vis) is produced.
'''

import bisect
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs
from ...utils.registry import register_dataset
from .seq2CondSplitPachifier import SPLIT_SHARDED_FORMAT
from .condPatchy import build_vis


class _ItemStore:
	'''Lazy sharded store (same layout as seq2CondPatchy._ItemStore, split format string).'''

	def __init__ (self, root):
		self.root = root
		self.artifact = torch.load(root, map_location='cpu', weights_only=False)
		assert self.artifact.get('format') == SPLIT_SHARDED_FORMAT, \
			f'unexpected format {self.artifact.get("format")!r} (want {SPLIT_SHARDED_FORMAT!r})'
		self._dir = os.path.dirname(root)
		self._shards = self.artifact['shards']
		self._offsets = []
		total = 0
		for shard in self._shards:
			self._offsets.append(total)
			total += shard['count']
		self._total = total
		self._cache = {}

	def __len__ (self):
		return self._total

	def _load_shard (self, shard_index):
		cached = self._cache.get(shard_index)
		if cached is None:
			path = os.path.join(self._dir, self._shards[shard_index]['file'])
			cached = torch.load(path, map_location='cpu', weights_only=False)['items']
			self._cache[shard_index] = cached
		return cached

	def get (self, index):
		shard_index = bisect.bisect_right(self._offsets, index) - 1
		local = index - self._offsets[shard_index]
		return self._load_shard(shard_index)[local]


_STORE_CACHE = {}


def _get_store (root):
	store = _STORE_CACHE.get(root)
	if store is None:
		store = _ItemStore(root)
		_STORE_CACHE[root] = store
	return store


@register_dataset
class Seq2CondSplitMidiPatchy (Dataset):
	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		splits = splits.split(':')

		def argi (i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		return tuple(
			cls(root, split, device=device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)

	def __init__ (self, root, split, device='cpu', shuffle=False, pad_id=0,
		w_midi=2, w_cross=2, w_midi_choices=None, w_cross_choices=None,
		patch_length=0, max_patches=0, random_crop=True, **_):
		super().__init__()
		self.device = device
		self.shuffle = shuffle
		self.pad_id = pad_id
		self.w_midi = w_midi
		self.w_cross = w_cross
		self.w_midi_choices = list(w_midi_choices) if w_midi_choices else None
		self.w_cross_choices = list(w_cross_choices) if w_cross_choices else None
		if patch_length:
			raise ValueError(
				'Seq2CondSplitMidiPatchy does not support patch_length (per-song measure coupling '
				'forbids cropping); set patch_length=0 and use max_patches.')
		self.patch_length = patch_length
		# max_patches bounds the MIDI patch count (Mp) per sample (the lyl prefix is unbounded but
		# trimmed to referenced measures). None/0 = no crop.
		self.max_patches = max_patches
		self.random_crop = random_crop
		self.store = _get_store(root)
		cfg = self.store.artifact.get('config', {})
		self.lyl_patch_size = int(cfg.get('lyl_patch_size', 16))
		self.midi_patch_size = int(cfg.get('midi_patch_size', cfg.get('patch_size', 64)))

		phases, cycle = parseFilterStr(split)
		self.indices = [i for i in range(len(self.store)) if i % cycle in phases]

	def __len__ (self):
		return len(self.indices)

	def _item (self, index):
		item = self.store.get(index)
		lyl_patches = item['lyl_patches'].long()			# [Lp, lyl_ps]
		midi_patches = item['midi_patches'].long()			# [Mp, midi_ps]
		lyl_meas = item['lyl_meas'].long()					# [Lp]
		midi_meas = item['midi_meas'].long()				# [Mp]
		midi_src = item['midi_src'].long()					# [Mp]
		drop_targets = []
		if self.max_patches and midi_patches.shape[0] > self.max_patches:
			lyl_patches, midi_patches, lyl_meas, midi_meas, midi_src, drop_targets = self._crop_midi(
				lyl_patches, midi_patches, lyl_meas, midi_meas, midi_src)
		return lyl_patches, midi_patches, lyl_meas, midi_meas, midi_src, drop_targets

	def _crop_midi (self, lyl_patches, midi_patches, lyl_meas, midi_meas, midi_src):
		'''Crop the MIDI body to a contiguous whole-measure window fitting max_patches (keeping the
		midi header patches, own_meas==0), then trim the lyl prefix to measures still referenced by
		the kept midi_src. Split-layout port of seq2CondPatchy._crop_midi — no m0/lyl_count juggling
		since the two modalities are already separate tensors.

		Returns the five cropped arrays + drop_targets (midi-frame positions whose supervised target
		was broken by the crop: the new body-start if a random head was dropped, and the last kept
		patch if the tail was clamped).
		'''
		Mp = midi_patches.shape[0]
		# midi header run: leading own_meas==0 patches (contiguous by construction).
		header_len = 0
		while header_len < Mp and int(midi_meas[header_len]) == 0:
			header_len += 1
		assert not bool((midi_meas[header_len:] == 0).any()), \
			'non-contiguous midi header patches (own_meas==0 after the header run)'
		budget = self.max_patches - header_len
		if budget <= 0:
			# header alone exceeds the cap: keep the first max_patches midi patches, trim lyl to all.
			keep = slice(0, self.max_patches)
			return lyl_patches, midi_patches[keep], lyl_meas, midi_meas[keep], midi_src[keep], []

		body_meas = midi_meas[header_len:]
		measures, last = [], None
		for mm in body_meas.tolist():
			assert last is None or mm >= last, 'non-monotonic midi body measure ids'
			if mm != last:
				measures.append(mm); last = mm
		counts = {int(mm): int((body_meas == mm).sum()) for mm in measures}
		if self.random_crop and len(measures) > 1:
			start_i = int(torch.randint(0, len(measures), (1,)))
		else:
			start_i = 0
		chosen, acc = [], 0
		for mm in measures[start_i:]:
			c = counts[mm]
			if chosen and acc + c > budget:
				break
			chosen.append(mm); acc += c
			if acc >= budget:
				break
		chosen_set = set(chosen)
		body_keep = torch.tensor([mm in chosen_set for mm in body_meas.tolist()], dtype=torch.bool)
		midi_keep = torch.cat((torch.ones(header_len, dtype=torch.bool), body_keep))
		# clamp to max_patches (measure grouping may slightly overshoot on the last measure).
		keep_idx = torch.nonzero(midi_keep, as_tuple=False).flatten()
		clamped = keep_idx.numel() > self.max_patches
		if clamped:
			keep_idx = keep_idx[:self.max_patches]
		midi_patches = midi_patches[keep_idx]
		midi_meas = midi_meas[keep_idx]
		midi_src = midi_src[keep_idx]

		# trim lyl prefix to measures referenced by the kept midi_src (keep header measure 0 always).
		max_kept_src = int(midi_src.max())
		assert bool((lyl_meas[1:] >= lyl_meas[:-1]).all()) if lyl_meas.numel() > 1 else True, \
			'non-monotonic lyl measure ids (tail trim assumes a non-decreasing prefix)'
		new_lyl_count = int((lyl_meas <= max_kept_src).sum())
		lyl_patches = lyl_patches[:new_lyl_count]
		lyl_meas = lyl_meas[:new_lyl_count]

		drop_targets = []
		if start_i > 0:
			drop_targets.append(header_len)			# new body start (cross-crop boundary)
		if clamped:
			drop_targets.append(int(keep_idx.numel()) - 1)
		return lyl_patches, midi_patches, lyl_meas, midi_meas, midi_src, drop_targets

	def __getitem__ (self, index):
		return self._item(self.indices[index])

	def __iter__ (self):
		indices = self.indices.copy()
		if self.shuffle:
			order = torch.randperm(len(indices)).tolist()
			indices = [indices[i] for i in order]
		for index in indices:
			yield self._item(index)

	def collateBatch (self, batch):
		B = len(batch)
		lyl_list = [ex[0] for ex in batch]
		midi_list = [ex[1] for ex in batch]
		Lmax = max(p.shape[0] for p in lyl_list)
		Mmax = max(p.shape[0] for p in midi_list)
		lyl_ps = lyl_list[0].shape[1]
		midi_ps = midi_list[0].shape[1]

		lyl_patches = pad_sequence(lyl_list, batch_first=True, padding_value=self.pad_id)		# [B,Lmax,lyl_ps]
		midi_patches = pad_sequence(midi_list, batch_first=True, padding_value=self.pad_id)		# [B,Mmax,midi_ps]
		lyl_masks = torch.zeros(B, Lmax, dtype=torch.long)
		lyl_meas = torch.zeros(B, Lmax, dtype=torch.long)
		midi_masks = torch.zeros(B, Mmax, dtype=torch.long)
		midi_meas = torch.zeros(B, Mmax, dtype=torch.long)
		midi_src = torch.zeros(B, Mmax, dtype=torch.long)
		midi_targets = torch.zeros(B, Mmax, dtype=torch.long)
		midi_positions = torch.zeros(B, Mmax, dtype=torch.long)
		midi_attn_mask = torch.zeros(B, 1, Mmax, Mmax, dtype=torch.bool)

		for b, (lylp, midip, lylm, midim, midis, drop_targets) in enumerate(batch):
			Lp = lylp.shape[0]
			Mp = midip.shape[0]
			lyl_masks[b, :Lp] = 1
			lyl_meas[b, :Lp] = lylm
			midi_masks[b, :Mp] = 1
			midi_meas[b, :Mp] = midim
			midi_src[b, :Mp] = midis
			midi_positions[b, :Mp] = torch.arange(Mp, dtype=torch.long)

			# supervision: every real midi patch is a next-patch target, except the FIRST midi patch
			# (cross-modality boundary) and any crop-broken positions — mirrors seq2CondPatchy.
			tgt = torch.ones(Mp, dtype=torch.long)
			tgt[0] = 0
			for pos in drop_targets:
				if 0 <= pos < Mp:
					tgt[pos] = 0
			midi_targets[b, :Mp] = tgt

			# midi->midi window mask: reuse build_vis on a reconstructed joint sequence, slice the
			# midi block. w_cross only affects the (discarded) cross block, so its value is immaterial.
			wm = self.w_midi
			if self.shuffle and self.w_midi_choices:
				wm = self.w_midi_choices[int(torch.randint(len(self.w_midi_choices), (1,)))]
			mod = torch.cat((torch.zeros(Lp, dtype=torch.long), torch.ones(Mp, dtype=torch.long)))
			own = torch.cat((lylm, midim))
			src = torch.cat((lylm, midis))
			vis = build_vis(mod, own, src, wm, self.w_cross)		# [Lp+Mp, Lp+Mp]
			midi_attn_mask[b, 0, :Mp, :Mp] = vis[Lp:, Lp:]
			# pad query rows attend self so SDPA softmax stays finite (outputs discarded).
			if Mp < Mmax:
				diag = torch.arange(Mp, Mmax)
				midi_attn_mask[b, 0, diag, diag] = True

		return dict(
			lyl_patches=lyl_patches.to(self.device),
			lyl_masks=lyl_masks.to(self.device),
			lyl_meas=lyl_meas.to(self.device),
			midi_patches=midi_patches.to(self.device),
			midi_masks=midi_masks.to(self.device),
			midi_meas=midi_meas.to(self.device),
			midi_src=midi_src.to(self.device),
			midi_targets=midi_targets.to(self.device),
			midi_positions=midi_positions.to(self.device),
			midi_attn_mask=midi_attn_mask.to(self.device),
			lyl_patch_size=self.lyl_patch_size,
			midi_patch_size=self.midi_patch_size,
		)
