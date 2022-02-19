
import os
import numpy as np
import random
import time
import logging
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from .utils import loadSplittedDatasets
from .score import makeReader, parseFilterStr
from .augmentor import Augmentor



def listAllImageNames (reader, filterStr, dir='/'):
	# split file name & ext name
	all_names = [os.path.splitext(name)[0] for name in reader.listFiles(dir)]

	if filterStr is None:
		return all_names

	phases, cycle = parseFilterStr(filterStr)

	return [name for i, name in enumerate(all_names) if (i % cycle) in phases]


class PerisData (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, labels, device='cpu', args_variant=None):
		return loadSplittedDatasets(cls, root=root, labels=labels, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, labels, label_fields, split='0/1', device='cpu', augmentor={}, shuffle=False, **_):
		self.reader, self.root = makeReader(root)
		self.shuffle = shuffle
		self.label_fields = label_fields
		self.device = device

		self.names = listAllImageNames(self.reader, split)

		dataframes = pd.read_csv(labels)
		self.labels = dict(zip(dataframes['hash'], dataframes.to_dict('records')))

		self.names = listAllImageNames(self.reader, split)
		self.names = [name for name in self.names if self.labels.get(name)]

		self.augmentor = Augmentor(augmentor, shuffle=self.shuffle)


	def __iter__ (self):
		if self.shuffle:
			random.shuffle(self.names)
			np.random.seed(int((time.time() * 1e+7 % 1e+7) + random.randint(0, 1e+5)))

		for i, name in enumerate(self.names):
			filename = f'{name}.jpg'
			if not self.reader.exists(filename):
				self.names.remove(name)
				logging.warn('image file missing, removed: %s', name)
				continue
			source = self.reader.readImage(filename)

			if source.shape[2] == 1:
				source = np.concatenate((source, source, source), axis=2)
			elif source.shape[2] > 3:
				source = source[:, :, :3]

			source = (source / 255.0).astype(np.float32)

			labels = self.labels[name]

			source, _ = self.augmentor.augment(source, None)

			yield source, labels


	def __len__ (self):
		return len(self.names)


	def collateBatch (self, batch):
		assert len(batch) == 1

		source, labels = batch[0]
		source = source.reshape((1,) + source.shape)
		source = torch.from_numpy(source).permute(0, 3, 1, 2).to(self.device)

		target = torch.tensor([[labels[field] for field in self.label_fields]], dtype=torch.float32).to(self.device)

		return source, target
