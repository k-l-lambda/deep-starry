'''
MidiPatchy — Dataset feeder for MIDI-text NotaGen/bGPT patch artifacts.

Counterpart of starry/lilylet/data/patchy.py for the MIDI-text modality. Consumes the
artifacts written by starry/midi/data/patchifier.py and yields the same batch contract
the shared two-level decoder expects (see starry.bgpt / LilyletNotaGen.forward):

	input_patches  LongTensor [B, T, patch_size]
	input_masks    LongTensor [B, T]   patch-level attention mask (1 real, 0 pad)
	input_targets  LongTensor [B, T]   supervision mask (1 = prediction target)

Two differences from LilyletPatchy, both consequences of MIDI songs being long and
header-only (no `%` style prompt, no `[field]` metadata):

  1. No prompt-dropout / prompt-vs-header split. The supervised region is everything
     after the leading <bos> patch (boundary = the <bos> index, normally 0).

  2. Random-crop at LOAD time. The packer stores each song's FULL patch sequence; here
     a `patch_length` window is cropped per access. With shuffle (train split) the start
     is random, so every epoch sees a different slice of each long song; without shuffle
     (val) the deterministic head window is used. The <bos> patch is preserved at the
     front of every crop so the token-level decoder always gets its boundary marker.
'''

import bisect
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs
from ...utils.registry import register_dataset


# Artifact `format` strings this store understands (single-file vs sharded).
_SHARDED_FORMAT = 'midi-notagen-patches-sharded'


class _ItemStore:
	'''Backing store for MidiPatchy items.

	Two layouts (mirrors the Lilylet store):
	  - single-file (version 1): root .pt holds `items` directly.
	  - sharded (version 2): root .pt is an index listing shard files; each shard is
	    loaded lazily on first access and cached in memory. Shared across the
	    train/val dataset instances of one config so shards load once, not per split.
	'''

	def __init__ (self, root):
		self.root = root
		self.artifact = torch.load(root, map_location='cpu')
		self.sharded = self.artifact.get('format') == _SHARDED_FORMAT

		if self.sharded:
			self._dir = os.path.dirname(root)
			self._shards = self.artifact['shards']
			self._offsets = []
			total = 0
			for shard in self._shards:
				self._offsets.append(total)
				total += shard['count']
			self._total = total
			self._cache = {}
		else:
			self._items = self.artifact['items']
			self._total = len(self._items)

	def __len__ (self):
		return self._total

	def _load_shard (self, shard_index):
		cached = self._cache.get(shard_index)
		if cached is None:
			path = os.path.join(self._dir, self._shards[shard_index]['file'])
			cached = torch.load(path, map_location='cpu')['items']
			self._cache[shard_index] = cached
		return cached

	def get (self, index):
		if not self.sharded:
			return self._items[index]
		shard_index = bisect.bisect_right(self._offsets, index) - 1
		local = index - self._offsets[shard_index]
		return self._load_shard(shard_index)[local]


# Cache stores across dataset instances built from the same root within a process.
_STORE_CACHE = {}


def _get_store (root):
	store = _STORE_CACHE.get(root)
	if store is None:
		store = _ItemStore(root)
		_STORE_CACHE[root] = store
	return store


@register_dataset
class MidiPatchy (Dataset):
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

	def __init__ (self, root, split, device='cpu', shuffle=False, pad_id=0, bos_id=1,
		patch_length=2048, **_):
		super().__init__()
		self.device = device
		self.shuffle = shuffle
		self.pad_id = pad_id
		self.bos_id = bos_id
		# Window cropped per access. <= 0 disables cropping (use the full song; only safe
		# for small corpora). Train split (shuffle) crops a random window; val the head.
		self.patch_length = patch_length
		self.store = _get_store(root)

		phases, cycle = parseFilterStr(split)
		self.indices = [i for i in range(len(self.store)) if i % cycle in phases]

	def __len__ (self):
		return len(self.indices)

	def _crop (self, patches):
		'''Crop `patches` to a patch_length window, keeping the <bos> patch at the front.

		patches[0] is the <bos> boundary marker (the packer always prepends it). For a
		random window starting at `s > 1`, we prepend patches[0] so the decoder still
		sees its boundary marker, then fill the rest with patches[s : s + patch_length-1].
		Shorter-than-window songs are returned whole.
		'''
		T = patches.shape[0]
		if self.patch_length <= 0 or T <= self.patch_length:
			return patches, 0

		body = T - 1					# patches after the leading <bos>
		win = self.patch_length - 1		# room for body after we re-prepend <bos>
		if self.shuffle:
			start = 1 + int(torch.randint(0, body - win + 1, ()).item())
		else:
			start = 1					# deterministic head window for val
		window = patches[start:start + win]
		cropped = torch.cat((patches[0:1], window), dim=0)
		return cropped, 0

	def _item (self, index):
		item = self.store.get(index)
		patches = item['patches'].long()
		patches, _ = self._crop(patches)
		# Supervision boundary: the <bos> patch (token[0] == bos_id) sits at the front.
		# Everything up to and INCLUDING <bos> is context; supervision begins after it.
		bos = (patches[:, 0] == self.bos_id).nonzero()
		boundary = int(bos[0].item()) if bos.numel() > 0 else 0
		# Attention mask: 1 for every real patch. Real padding (and its 0s) is only
		# introduced at batch time by collateBatch.
		mask = torch.ones(patches.shape[0], dtype=torch.long)
		return patches, mask, boundary

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
		input_patches = [ex[0] for ex in batch]
		input_masks = [ex[1] for ex in batch]
		# Supervision mask: copy the attention mask, then zero the first boundary+1 patches
		# (the <bos> boundary) so they are attended but never prediction targets. Padding
		# stays 0 after pad_sequence.
		input_targets = []
		for (_, m, boundary) in batch:
			t = m.clone()
			t[:boundary + 1] = 0
			input_targets.append(t)
		input_patches = pad_sequence(input_patches, batch_first=True, padding_value=self.pad_id)
		input_masks = pad_sequence(input_masks, batch_first=True, padding_value=0)
		input_targets = pad_sequence(input_targets, batch_first=True, padding_value=0)
		return dict(
			input_patches=input_patches.to(self.device),
			input_masks=input_masks.to(self.device),
			input_targets=input_targets.to(self.device),
		)
