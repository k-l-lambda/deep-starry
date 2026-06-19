import bisect
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs
from ...utils.registry import register_dataset


class _ItemStore:
	'''Backing store for LilyletPatchy items.

	Two layouts:
	  - single-file (version 1): root .pt holds `items` directly.
	  - sharded (version 2): root .pt is an index listing shard files; each shard is
	    loaded lazily on first access and cached in memory (the host has ample RAM,
	    so over one epoch the whole corpus ends up resident, but startup is instant
	    and we never hold more than the touched shards).

	The store is shared across the train/val dataset instances of one config so shards
	are loaded and cached once, not per split.
	'''

	def __init__(self, root):
		self.root = root
		self.artifact = torch.load(root, map_location='cpu')
		self.sharded = self.artifact.get('format') == 'lilylet-notagen-patches-sharded'

		if self.sharded:
			self._dir = os.path.dirname(root)
			self._shards = self.artifact['shards']
			# cumulative start offset of each shard -> global item index space
			self._offsets = []
			total = 0
			for shard in self._shards:
				self._offsets.append(total)
				total += shard['count']
			self._total = total
			self._cache = {}  # shard_index -> list[item]
		else:
			self._items = self.artifact['items']
			self._total = len(self._items)

	def __len__(self):
		return self._total

	def _load_shard(self, shard_index):
		cached = self._cache.get(shard_index)
		if cached is None:
			path = os.path.join(self._dir, self._shards[shard_index]['file'])
			cached = torch.load(path, map_location='cpu')['items']
			self._cache[shard_index] = cached
		return cached

	def get(self, index):
		if not self.sharded:
			return self._items[index]
		# locate the shard whose range contains `index`
		shard_index = bisect.bisect_right(self._offsets, index) - 1
		local = index - self._offsets[shard_index]
		return self._load_shard(shard_index)[local]


# Cache stores across dataset instances built from the same root within a process.
_STORE_CACHE = {}


def _get_store(root):
	store = _STORE_CACHE.get(root)
	if store is None:
		store = _ItemStore(root)
		_STORE_CACHE[root] = store
	return store


@register_dataset
class LilyletPatchy(Dataset):
	@classmethod
	def load(cls, root, args, splits, device='cpu', args_variant=None, **_):
		splits = splits.split(':')

		def argi(i):
			if args_variant is None:
				return args
			return mergeArgs(args, args_variant.get(i))

		return tuple(
			cls(root, split, device=device, shuffle='*' in split, **argi(i))
			for i, split in enumerate(splits)
		)

	def __init__(self, root, split, device='cpu', shuffle=False, pad_id=0, bos_id=1, prompt_dropout=0.0, **_):
		super().__init__()
		self.device = device
		self.shuffle = shuffle
		self.pad_id = pad_id
		self.bos_id = bos_id
		# Probability of independently dropping each leading `%`-style prompt comment line
		# (style conditioning augmentation). 0 = never drop (default).
		self.prompt_dropout = prompt_dropout
		self.store = _get_store(root)

		phases, cycle = parseFilterStr(split)
		self.indices = [
			i for i in range(len(self.store))
			if i % cycle in phases
		]

	def __len__(self):
		return len(self.indices)

	# The `%` mark that begins a prompt comment line, as a token id (ASCII '%').
	PROMPT_MARK_ID = ord('%')

	def _drop_prompt_lines(self, patches, boundary):
		'''Randomly drop whole `%`-style prompt comment lines from patches[:boundary].

		A comment line is a `%`-starting patch plus the following non-`%` continuation
		patches that belong to it (a single line can span >1 patch). Each line is dropped
		independently with probability self.prompt_dropout; dropping a line removes its
		`%` patch AND its continuation patches. The <bos> patch at `boundary` and the
		supervised body after it are untouched.
		'''
		if self.prompt_dropout <= 0 or boundary <= 0:
			return patches, boundary

		# group the prompt patches [0, boundary) into comment lines: each group starts at a
		# `%`-marked patch and runs until the next `%`-marked patch (or the boundary).
		starts = [i for i in range(boundary) if int(patches[i, 0]) == self.PROMPT_MARK_ID]
		if not starts:
			return patches, boundary
		bounds = starts + [boundary]

		keep = torch.ones(patches.shape[0], dtype=torch.bool)
		# any leading patches before the first `%` (shouldn't normally exist) are kept.
		for g in range(len(starts)):
			if torch.rand(()).item() < self.prompt_dropout:
				keep[bounds[g]:bounds[g + 1]] = False

		if bool(keep.all()):
			return patches, boundary

		dropped_before_boundary = int((~keep[:boundary]).sum().item())
		return patches[keep], boundary - dropped_before_boundary

	def _item(self, index):
		item = self.store.get(index)
		patches = item['patches'].long()
		# Supervision boundary: the `<bos>` patch (token[0] == bos_id) sits after the
		# leading `%`-style prompt patches and before the first `[`-header. Everything up
		# to and INCLUDING <bos> is prompt/context (not a prediction target); supervision
		# begins at the patch after <bos>. Old-format items have <bos> at index 0, so
		# boundary 0 reproduces the previous supervise-all-but-first behavior.
		bos = (patches[:, 0] == self.bos_id).nonzero()
		boundary = int(bos[0].item()) if bos.numel() > 0 else 0
		# Optionally drop whole prompt comment lines (style-conditioning augmentation).
		patches, boundary = self._drop_prompt_lines(patches, boundary)
		# Attention mask: 1 for every real patch (incl. the unsupervised prompt). Real
		# padding (and its 0s) is only introduced at batch time by collateBatch.
		mask = torch.ones(patches.shape[0], dtype=torch.long)
		return patches, mask, boundary

	def __getitem__(self, index):
		return self._item(self.indices[index])

	def __iter__(self):
		indices = self.indices.copy()
		if self.shuffle:
			order = torch.randperm(len(indices)).tolist()
			indices = [indices[i] for i in order]
		for index in indices:
			yield self._item(index)

	def collateBatch(self, batch):
		input_patches = [ex[0] for ex in batch]
		input_masks = [ex[1] for ex in batch]
		# Supervision mask: copy the attention mask, then zero the first boundary+1 patches
		# (the prompt + the <bos> boundary) so they are attended but never prediction
		# targets. Padding stays 0 after pad_sequence.
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
