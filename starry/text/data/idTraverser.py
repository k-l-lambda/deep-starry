
import numpy as np
import torch
from torch.utils.data import IterableDataset

from ...utils.parsers import parseFilterStr



class IdTraverser (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		splits = splits.split(':')

		return (
			cls(split, device, shuffle='*' in split, **args)
			for i, split in enumerate(splits)
		)


	def __init__ (self, split, device, shuffle, vocab_size, **_):
		super().__init__()

		self.device = device
		self.shuffle = shuffle

		phases, cycle = parseFilterStr(split)

		self.ids = [id for id in range(vocab_size) if id % cycle in phases]


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.ids)

		for id in self.ids:
			yield id


	def __len__ (self):
		return len(self.ids)


	def collateBatch (self, batch):
		return torch.tensor(batch, dtype=torch.long).to(self.device)
