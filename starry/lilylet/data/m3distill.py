import os
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ...utils.parsers import parseFilterStr, mergeArgs


# Cache stores across dataset instances built from the same root within a process,
# so the train/val splits of one config share a single in-memory artifact load.
_STORE_CACHE = {}


class _ItemStore:
	'''Backing store for the lilylet-m3-abc-pooled artifact (single-file, version 1).

	The artifact (produced by tools/lilylet/preprocessLilyletM3.py) holds an `items`
	list; each item pairs a Lilylet patch tensor with the mean-pooled ABC M3 teacher
	embedding:
	  - patches:      uint8 [P, patch_size]   Lilylet tokenize+patchize (style prompt dropped)
	  - m3_embedding: float16 [hidden]        masked-mean-pooled ABC M3 vector (teacher target)
	  - m3_patches:   int                     original pre-pool patch count (metadata)
	'''

	def __init__ (self, root):
		self.root = root
		self.artifact = torch.load(root, map_location='cpu')
		self._items = self.artifact['items']
		self._total = len(self._items)
		self.config = self.artifact.get('config', {})

	def __len__ (self):
		return self._total

	def get (self, index):
		return self._items[index]


def _get_store (root):
	store = _STORE_CACHE.get(root)
	if store is None:
		store = _ItemStore(root)
		_STORE_CACHE[root] = store
	return store


class LilyletM3Distill (Dataset):
	'''Feeder for ABC→Lilylet M3 distillation.

	Yields, per example, the Lilylet patch sequence (student input) and the
	mean-pooled ABC M3 embedding (frozen-teacher regression target). The student
	encodes the patches and is trained so its pooled output matches the target.
	'''

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
		max_patches=2048, random_truncate=None, **_):
		super().__init__()
		self.device = device
		self.shuffle = shuffle
		self.pad_id = pad_id
		# Cap the Lilylet patch sequence length. Over-long pieces are truncated with a
		# random head/tail/middle window (mirroring CLaMP's M3Patchilizer truncation) when
		# random_truncate is on, else a deterministic head cut. random_truncate defaults to
		# the split's shuffle flag (train shuffles → random aug; val is deterministic head).
		self.max_patches = max_patches
		self.random_truncate = shuffle if random_truncate is None else random_truncate
		self.store = _get_store(root)

		phases, cycle = parseFilterStr(split)
		self.indices = [i for i in range(len(self.store)) if i % cycle in phases]

	def __len__ (self):
		return len(self.indices)

	def _truncate (self, patches):
		'''Cap to max_patches with a head/tail/middle window (CLaMP-style). Random window
		when self.random_truncate, else head. Returns the (possibly unchanged) patches.'''
		n = patches.shape[0]
		cap = self.max_patches
		if not cap or n <= cap:
			return patches
		if not self.random_truncate:
			return patches[:cap]
		choice = random.choice(('head', 'tail', 'middle'))
		if choice == 'head':
			return patches[:cap]
		if choice == 'tail':
			return patches[-cap:]
		start = random.randint(1, n - cap)
		return patches[start:start + cap]

	def _item (self, index):
		item = self.store.get(index)
		patches = self._truncate(item['patches'].long())      # [P', patch_size], P' <= max_patches
		target = item['m3_embedding'].float()                  # [hidden]
		mask = torch.ones(patches.shape[0], dtype=torch.long)  # 1 per real patch
		return patches, mask, target

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
		input_patches = pad_sequence([ex[0] for ex in batch], batch_first=True, padding_value=self.pad_id)
		input_masks = pad_sequence([ex[1] for ex in batch], batch_first=True, padding_value=0)
		target_embedding = torch.stack([ex[2] for ex in batch])  # [B, hidden]
		return dict(
			input_patches=input_patches.to(self.device),
			input_masks=input_masks.to(self.device),
			target_embedding=target_embedding.to(self.device),
		)
