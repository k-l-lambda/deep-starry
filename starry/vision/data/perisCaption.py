
import time
import numpy as np
import random
import logging
import pandas as pd
from torch.utils.data import IterableDataset

from .utils import loadSplittedDatasets, listAllImageNames
from .score import makeReader



class PerisCaption (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, labels, device='cpu', args_variant=None):
		return loadSplittedDatasets(cls, root=root, labels=labels, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, labels, label_fields, split='0/1', device='cpu', augmentor={}, shuffle=False, **_):
		self.reader, self.root = makeReader(root)
		self.shuffle = shuffle
		self.label_fields = label_fields
		self.device = device

		dataframes = pd.read_csv(labels)
		self.labels = dict(zip(dataframes['hash'], dataframes.to_dict('records')))

		self.names = listAllImageNames(self.reader, split)
		self.names = [name for name in self.names if self.labels.get(name)]


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
			if source is None:
				self.names.remove(name)
				logging.warn('image reading failed, removed: %s', name)
				continue

			if len(source.shape) < 3:
				source = source.reshape(source.shape + (1,))
			if source.shape[2] == 1:
				source = np.concatenate((source, source, source), axis=2)
			elif source.shape[2] > 3:
				source = source[:, :, :3]

			source = (source / 255.0).astype(np.float32)

			labels = self.labels[name]

			yield source, labels


	def __len__ (self):
		return len(self.names)
