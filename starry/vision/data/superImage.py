
import random
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from .utils import loadSplittedDatasets, listAllImageNames
from .imageReader import makeReader


def collateBatchSingle (batch, device):
	assert len(batch) == 1

	x, y = batch[0]

	return x.to(device), y.to(device)


class DimensionCluster:
	def __init__ (self, csv_file, cluster_size=0x100000, no_repeat=False, shuffle=False):
		dataframes = pd.read_csv(csv_file)

		self.name_dict = {}
		self.shuffle = shuffle

		sizes = set(dataframes.loc[:, 'size'])
		for size in sizes:
			w, h = map(int, size.split('x'))
			n_img = max(1, cluster_size // (w * h))

			names = dataframes[dataframes['size'] == size]['name']
			names_repeat = list(names) * n_img
			if no_repeat:
				names = list(names)
				for i in range(0, len(names), n_img):
					self.name_dict[size] = names[i:i + n_img]
			else:
				for i, name in enumerate(names):
					self.name_dict[size] = names_repeat[i:i + n_img]


	def __iter__ (self):
		items = list(self.name_dict.items())
		if self.shuffle:
			random.shuffle(items)

		for size, names in items:
			w, h = map(int, size.split('x'))
			yield (h, w), names


class SuperImage (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		return loadSplittedDatasets(cls, root=root, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, split='0/1', dimensions=None, cluster_size=0x100000, device='cpu', shuffle=False, **_):
		self.reader, _ = makeReader(root)
		self.shuffle = shuffle
		self.device = device

		self.cluster = DimensionCluster(dimensions, cluster_size=cluster_size, no_repeat=not shuffle, shuffle=shuffle)


	def __iter__ (self):
		for (h, w), names in self.cluster:
			yield (h, w), names


	def __len__ (self):
		return len(self.names)


	def collateBatch (self, batch):
		return collateBatchSingle(batch, self.device)
