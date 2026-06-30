'''CondMidiPatchy — feeder for the conditioned-MIDI measurewise joint-sequence dataset.

Consumes the artifact written by starry.midi.data.condPatchifier.pack: each item is ONE
joint sequence [lilylet score patches] ++ [midi event patches], with per-patch modality and
measure indices. This feeder builds, at batch time, the custom windowed/block attention mask
that couples midi measures to lilylet measures, and the supervision mask that trains ONLY the
midi segment (lilylet is a read-only condition).

Batch contract (consumed by starry.midi.models.condBgpt.CondMidiBGPT):
	input_patches   LongTensor [B, T, patch_size]   token ids (lilylet ids then midi ids)
	input_masks     LongTensor [B, T]               1 = real patch, 0 = padding
	input_targets   LongTensor [B, T]               1 = supervised next-patch target (midi only)
	input_positions LongTensor [B, T]               0..T-1 per sample (padded 0)
	attn_mask       BoolTensor [B, 1, T, T]          True = query may attend key (custom mask)
	modality        LongTensor [B, T]               0 = lilylet, 1 = midi
	lyl_counts      LongTensor [B]                   lilylet-segment length per sample

The 4D bool attn_mask is passed straight through to the HF Llama backbone (transformers
>=4.53 returns a 4D attention_mask as-is; SDPA reads bool True=attend).
'''

import bisect
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs
from ...utils.registry import register_dataset
from .condPatchifier import SHARDED_FORMAT


class _ItemStore:
	'''Lazy sharded store (mirrors starry.midi.data.patchy._ItemStore, new format string).'''

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


def build_vis (modality, own_meas, src_meas, w_midi, w_cross):
	'''Build the [T, T] boolean visibility mask (True = query q may attend key k).

	modality  [T] long  0 = lilylet, 1 = midi
	own_meas  [T] long  patch's own measure (lyl j / midi i; 0 = prefix/header)
	src_meas  [T] long  lilylet-aligned measure (lyl j / midi source_measure(i); 0 = prefix/header)
	w_midi    int       midi->midi window in MEASURES
	w_cross   int       midi->lilylet window in MEASURES

	Rules (all under the global causal floor k <= q):
	  lyl -> lyl   : full causal
	  lyl -> midi  : never (also forbidden by causal since midi is later)
	  midi -> lyl  : j == 0 (lyl prompt/header) -> always; i == 0 (midi header) -> never;
	                 else (i >= j) and (i < j + w_cross), with i = src[q], j = src[k]
	  midi -> midi : header (own 0) -> full causal within header; else c in [a-w_midi+1, a],
	                 with a = own[q], c = own[k]
	  eom patches  : carry a normal own/src measure, so they fall through the midi rules with
	                 NO special-casing.
	'''
	T = modality.shape[0]
	dev = modality.device
	idx = torch.arange(T, device=dev)
	causal = idx[:, None] >= idx[None, :]				# [T,T] k<=q

	q_lyl = (modality == 0)[:, None].expand(T, T)
	k_lyl = (modality == 0)[None, :].expand(T, T)
	q_midi = ~q_lyl
	k_midi = ~k_lyl

	own_q = own_meas[:, None].expand(T, T)
	own_k = own_meas[None, :].expand(T, T)
	src_q = src_meas[:, None].expand(T, T)
	src_k = src_meas[None, :].expand(T, T)

	vis = torch.zeros(T, T, dtype=torch.bool, device=dev)

	# lyl -> lyl : full causal
	vis |= q_lyl & k_lyl & causal

	# midi -> lyl : cross block by lilylet-aligned measure
	cross = q_midi & k_lyl & causal
	j_zero = src_k == 0								# lyl prompt/header: global condition
	i_zero = src_q == 0								# midi header: sees nothing on the lyl side
	in_window = (src_q >= src_k) & (src_q < src_k + w_cross)
	cross_vis = torch.where(j_zero, torch.ones_like(vis),
		torch.where(i_zero, torch.zeros_like(vis), in_window))
	vis |= cross & cross_vis

	# midi -> midi : windowed causal by midi measure
	mm = q_midi & k_midi & causal
	header = (own_q == 0) | (own_k == 0)			# midi header region: full causal
	win = (own_k > own_q - w_midi) & (own_k <= own_q)
	vis |= mm & (header | win)

	# lyl -> midi stays False (causal already forbids it).
	return vis


@register_dataset
class CondMidiPatchy (Dataset):
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
		w_midi=4, w_cross=2, patch_length=0, **_):
		super().__init__()
		self.device = device
		self.shuffle = shuffle
		self.pad_id = pad_id
		self.w_midi = w_midi
		self.w_cross = w_cross
		# patch_length kept for API parity; per-song coupling forbids cropping, so 0 (no crop)
		# is the only supported value — a crop would sever the measure alignment.
		self.patch_length = patch_length
		self.store = _get_store(root)

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
		return patches, modality, own_meas, src_meas, lyl_count

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

		for b, (patches, mod, own_meas, src_meas, lyl_count) in enumerate(batch):
			T = patches.shape[0]
			input_masks[b, :T] = 1
			modality[b, :T] = mod
			input_positions[b, :T] = torch.arange(T, dtype=torch.long)
			lyl_counts[b] = lyl_count

			# Supervision: ONLY the midi segment is a prediction target. The very first midi
			# patch (index lyl_count) is NOT a target — we never force the lilylet segment's
			# last hidden state to emit the first midi patch across the modality boundary
			# (at generation the deterministic midi header is fed as a prompt instead).
			tgt = (mod == 1).long()
			tgt[0] = 0					# defensive (lyl_count >= 1 always, so index 0 is lyl)
			if lyl_count < T:
				tgt[lyl_count] = 0		# drop the cross-boundary prediction
			input_targets[b, :T] = tgt

			vis = build_vis(mod, own_meas, src_meas, self.w_midi, self.w_cross)
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
		)


