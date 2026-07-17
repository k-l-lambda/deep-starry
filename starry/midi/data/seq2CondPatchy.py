'''Seq2CondMidiPatchy — feeder for the conditioned-MIDI measurewise midiseq2 dataset.

Sibling of starry.midi.data.condPatchy (CondMidiPatchy), consuming the artifact written by
starry.midi.data.seq2CondPachifier.pack. Same joint-sequence idea — [lilylet score patches]
++ [midi patches] with the windowed/block measure-coupling attention mask (build_vis, reused
verbatim from condPatchy) — but the MIDI side is midiseq2 measure-patches (patch_size 64, one
measure's token run chunked to whole patches, <eom> TOKEN terminated) rather than the old
one-event-per-patch scheme.

Differences from CondMidiPatchy:
  - Patches are int16 (midiseq2 ids reach 837), stored in a patch_size=64 frame. Lilylet
    patches occupy columns [0, lyl_patch_size=16) and are <pad> in the tail; midi patches use
    the full width. The `lyl_patch_size` split is read from the artifact and exposed on the
    batch so the model can slice the lilylet columns for its frozen encoder.
  - No standalone <eom> patch: bar structure is carried by the <eom> token inside each measure's
    last patch, so cropping / measure grouping keys off the `measures` tag exactly as before.

Batch contract (mirrors CondMidiPatchy; consumed by a midiseq2 joint decoder model):
	input_patches   LongTensor [B, T, patch_size]   token ids (lyl ids in cols<16, midi ids full)
	input_masks     LongTensor [B, T]               1 = real patch, 0 = padding
	input_targets   LongTensor [B, T]               1 = supervised next-patch target (midi only)
	input_positions LongTensor [B, T]               per-modality 0..n-1
	attn_mask       BoolTensor [B, 1, T, T]          joint windowed/block mask (True = attend)
	modality        LongTensor [B, T]               0 = lilylet, 1 = midi
	lyl_counts      LongTensor [B]                   lilylet-segment length per sample
	lyl_patch_size  int                              encoder-slice width (16)
'''

import bisect
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs
from ...utils.registry import register_dataset
from .seq2CondPachifier import SHARDED_FORMAT
from .condPatchy import build_vis


class _ItemStore:
	'''Lazy sharded store (same layout as condPatchy._ItemStore, new format string).'''

	def __init__ (self, root):
		self.root = root
		self.artifact = torch.load(root, map_location='cpu', weights_only=False)
		assert self.artifact.get('format') == SHARDED_FORMAT, \
			f'unexpected format {self.artifact.get("format")!r} (want {SHARDED_FORMAT!r})'
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
class Seq2CondMidiPatchy (Dataset):
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
		# Random-window augmentation (train only), identical semantics to CondMidiPatchy.
		self.w_midi_choices = list(w_midi_choices) if w_midi_choices else None
		self.w_cross_choices = list(w_cross_choices) if w_cross_choices else None
		# patch_length is UNSUPPORTED (per-song measure coupling forbids cropping the joint
		# sequence); use max_patches to bound T by cropping only the midi body to whole measures.
		if patch_length:
			raise ValueError(
				'Seq2CondMidiPatchy does not support patch_length (per-song measure coupling '
				'forbids cropping the joint sequence); set patch_length=0 and use max_patches.')
		self.patch_length = patch_length
		self.max_patches = max_patches
		self.random_crop = random_crop
		self.store = _get_store(root)
		# encoder-slice width baked into the artifact (columns [0, lyl_patch_size) hold lilylet ids).
		self.lyl_patch_size = int(self.store.artifact.get('config', {}).get('lyl_patch_size', 16))

		phases, cycle = parseFilterStr(split)
		self.indices = [i for i in range(len(self.store)) if i % cycle in phases]

	def __len__ (self):
		return len(self.indices)

	def _item (self, index):
		item = self.store.get(index)
		patches = item['patches'].long()				# [T, patch_size]
		modality = item['modality'].long()				# [T]
		own_meas = item['measures'].long()				# [T]
		src_meas = item['src_measures'].long()			# [T]
		lyl_count = int(item['lyl_count'])
		drop_targets = []
		if self.max_patches and patches.shape[0] > self.max_patches:
			patches, modality, own_meas, src_meas, lyl_count, drop_targets = self._crop_midi(
				patches, modality, own_meas, src_meas, lyl_count)
		return patches, modality, own_meas, src_meas, lyl_count, drop_targets

	def _crop_midi (self, patches, modality, own_meas, src_meas, lyl_count):
		'''Crop the MIDI body to a contiguous whole-measure window so the joint length fits
		max_patches (keeps the lilylet prefix + midi header intact). Identical measure-grouping
		logic to CondMidiPatchy._crop_midi — the only midi-side difference (midiseq2 vs event
		patches) does not touch the measure tags this operates on.'''
		T = patches.shape[0]
		is_midi = modality == 1
		midi_pos = torch.nonzero(is_midi, as_tuple=False).flatten()
		if midi_pos.numel() == 0:
			return patches, modality, own_meas, src_meas, lyl_count, []
		m0 = int(midi_pos[0])
		midi_own = own_meas[m0:]
		header_len = 0
		while header_len < midi_own.numel() and int(midi_own[header_len]) == 0:
			header_len += 1
		assert not bool((midi_own[header_len:] == 0).any()), \
			'non-contiguous midi header patches (own_meas==0 after the header run)'
		body_start = m0 + header_len
		fixed = body_start
		budget = self.max_patches - fixed
		if budget <= 0:
			keep = slice(0, self.max_patches)
			return (patches[keep], modality[keep], own_meas[keep], src_meas[keep],
				min(lyl_count, self.max_patches), [])
		body_own = own_meas[body_start:]
		measures, last = [], None
		for mm in body_own.tolist():
			assert last is None or mm >= last, 'non-monotonic midi body measure ids'
			if mm != last:
				measures.append(mm); last = mm
		counts = {int(mm): int((body_own == mm).sum()) for mm in measures}
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
		body_keep = torch.tensor([mm in chosen_set for mm in body_own.tolist()], dtype=torch.bool)
		midi_keep_rel = torch.cat((torch.ones(header_len, dtype=torch.bool), body_keep))
		max_kept_src = int(src_meas[m0:][midi_keep_rel].max())
		lyl_src = src_meas[:m0]
		assert lyl_src.numel() == 0 or bool((lyl_src[1:] >= lyl_src[:-1]).all()), \
			'non-monotonic lilylet src_meas (tail trim assumes a non-decreasing prefix)'
		new_lyl_count = int((lyl_src <= max_kept_src).sum())
		keep_mask = torch.zeros(T, dtype=torch.bool)
		keep_mask[:new_lyl_count] = True
		keep_mask[m0:body_start] = True
		keep_mask[body_start:] = body_keep
		keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
		cropped_body_start = new_lyl_count + header_len
		clamped = keep_idx.numel() > self.max_patches
		if clamped:
			keep_idx = keep_idx[:self.max_patches]
		drop_targets = []
		if start_i > 0:
			drop_targets.append(cropped_body_start)
		if clamped:
			drop_targets.append(int(keep_idx.numel()) - 1)
		return (patches[keep_idx], modality[keep_idx], own_meas[keep_idx],
			src_meas[keep_idx], new_lyl_count, drop_targets)

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
		patches_list = [ex[0] for ex in batch]
		B = len(batch)
		Tmax = max(p.shape[0] for p in patches_list)
		ps = patches_list[0].shape[1]

		input_patches = pad_sequence(patches_list, batch_first=True, padding_value=self.pad_id)	# [B,Tmax,ps]
		input_masks = torch.zeros(B, Tmax, dtype=torch.long)
		modality = torch.zeros(B, Tmax, dtype=torch.long)
		input_targets = torch.zeros(B, Tmax, dtype=torch.long)
		input_positions = torch.zeros(B, Tmax, dtype=torch.long)
		lyl_counts = torch.zeros(B, dtype=torch.long)
		attn_mask = torch.zeros(B, 1, Tmax, Tmax, dtype=torch.bool)

		for b, (patches, mod, own_meas, src_meas, lyl_count, drop_targets) in enumerate(batch):
			T = patches.shape[0]
			input_masks[b, :T] = 1
			modality[b, :T] = mod
			# Positions restart per modality: lilylet 0..L-1, midi 0..Mp-1 (own RoPE frame each).
			input_positions[b, :lyl_count] = torch.arange(lyl_count, dtype=torch.long)
			input_positions[b, lyl_count:T] = torch.arange(T - lyl_count, dtype=torch.long)
			lyl_counts[b] = lyl_count

			# Supervision: only the midi segment; drop the first midi patch (cross-boundary) and
			# any crop-broken targets, identical to CondMidiPatchy.
			tgt = (mod == 1).long()
			tgt[0] = 0
			if lyl_count < T:
				tgt[lyl_count] = 0
			for pos in drop_targets:
				if 0 <= pos < T:
					tgt[pos] = 0
			input_targets[b, :T] = tgt

			# Per-sample window: random augmentation on train (if *_choices), else fixed w_*.
			wm, wc = self.w_midi, self.w_cross
			if self.shuffle and self.w_midi_choices:
				wm = self.w_midi_choices[int(torch.randint(len(self.w_midi_choices), (1,)))]
			if self.shuffle and self.w_cross_choices:
				wc = self.w_cross_choices[int(torch.randint(len(self.w_cross_choices), (1,)))]
			vis = build_vis(mod, own_meas, src_meas, wm, wc)
			attn_mask[b, 0, :T, :T] = vis
			# pad query rows attend self so SDPA softmax stays finite (outputs discarded).
			if T < Tmax:
				diag = torch.arange(T, Tmax)
				attn_mask[b, 0, diag, diag] = True

		return dict(
			input_patches=input_patches.to(self.device),
			input_masks=input_masks.to(self.device),
			input_targets=input_targets.to(self.device),
			input_positions=input_positions.to(self.device),
			attn_mask=attn_mask.to(self.device),
			modality=modality.to(self.device),
			lyl_counts=lyl_counts.to(self.device),
			lyl_patch_size=self.lyl_patch_size,
		)
