'''`loadDataset` must shuffle the SAMPLER of a map-style dataset.

Every map-style dataset class in this repo sets `self.shuffle` from the split's '*' and then shuffles
inside its own `__iter__`. DataLoader reaches a map-style dataset through `__getitem__` and never calls
`__iter__`, so before the fix that shuffle was dead code and the train split was served in file order
on every epoch. These tests pin the sampler choice, which is the thing that was silently wrong.
'''

import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import RandomSampler, SequentialSampler

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.utils.registry import DATASETS


class _MapFake (Dataset):
	'''Mirrors the repo's map-style contract: shuffle from the split, a shuffling __iter__, collateBatch.'''

	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		return tuple(cls(shuffle='*' in split) for split in splits.split(':'))

	def __init__ (self, shuffle=False, n=8):
		self.shuffle = shuffle
		self.n = n

	def __len__ (self):
		return self.n

	def __getitem__ (self, index):
		return index

	def __iter__ (self):
		# Present, and irrelevant to DataLoader -- exactly the trap this suite guards.
		order = torch.randperm(self.n).tolist() if self.shuffle else list(range(self.n))
		return iter(order)

	def collateBatch (self, batch):
		return list(batch)


class _IterableFake (IterableDataset):
	'''An IterableDataset keeps its own __iter__; DataLoader rejects shuffle=True for one.'''

	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		return tuple(cls(shuffle='*' in split) for split in splits.split(':'))

	def __init__ (self, shuffle=False, n=8):
		self.shuffle = shuffle
		self.n = n

	def __iter__ (self):
		return iter(range(self.n))

	def collateBatch (self, batch):
		return list(batch)


def _config (type_name):
	# Direct constructor: `data` given inline makes it a created config, and volatile keeps it off disk.
	return Configuration('.', {
		'data': {'type': type_name, 'root': '.', 'batch_size': 2, 'splits': '*0/2:1/2', 'args': {}},
	}, volatile=True)


def _load (type_name, cls):
	DATASETS[type_name] = cls
	try:
		return loadDataset(_config(type_name), data_dir='.')
	finally:
		DATASETS.pop(type_name, None)


def test_map_style_train_split_gets_a_random_sampler ():
	train, val = _load('_MapFake', _MapFake)
	assert isinstance(train.sampler, RandomSampler), \
		'a map-style train split ("*") must be sampled randomly, not in file order'
	assert isinstance(val.sampler, SequentialSampler), \
		'val must stay deterministic so epochs remain comparable'


def test_map_style_train_order_varies_between_passes ():
	train, _ = _load('_MapFake', _MapFake)
	torch.manual_seed(0)
	passes = {tuple(sum((b for b in train), [])) for _ in range(6)}
	assert len(passes) > 1, 'two passes over the train split produced identical order'


def test_iterable_dataset_is_not_given_shuffle ():
	# The regression this guards: passing shuffle=True to an IterableDataset raises ValueError.
	train, val = _load('_IterableFake', _IterableFake)
	for loader in (train, val):
		assert loader.sampler is None or not isinstance(loader.sampler, RandomSampler)
		assert list(sum((b for b in loader), [])) == list(range(8))
