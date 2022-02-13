
import os
import numpy as np
import random
import time
import cv2
import logging
import pandas as pd

from .utils import loadSplittedDatasets
from .score import makeReader, parseFilterStr



def listAllImageNames (reader, filterStr, dir='/'):
	# split file name & ext name
	all_names = [os.path.splitext(name)[0] for name in reader.listFiles(dir)]

	if filterStr is None:
		return all_names

	phases, cycle = parseFilterStr(filterStr)

	return [name for i, name in enumerate(all_names) if (i % cycle) in phases]


class PerisData:
	@classmethod
	def load (cls, root, args, splits, args_variant=None):
		return loadSplittedDatasets(cls, root=root, args=args, splits=splits, args_variant=args_variant)


	def __init__ (self, root, labels, split='0/1', shuffle=False, **_):
		self.reader, self.root = makeReader(root)
		self.shuffle = shuffle

		self.names = listAllImageNames(self.reader, split)

		dataframes = pd.read_csv(labels)
		self.labels = dict(zip(dataframes['hash'], dataframes.to_dict('records')))


	def __iter__ (self):
		if self.shuffle:
			random.shuffle(self.names)
			np.random.seed(int((time.time() * 1e+7 % 1e+7) + random.randint(0, 1e+5)))

		while True:
			for i, name in enumerate(self.names):
				filename = f'{name}.jpg'
				if not self.reader.exists(filename):
					self.names.remove(name)
					logging.warn('image file missing, removed: %s', name)
					continue
				source = self.reader.readImage(filename)

				#source = (source / 255.0).astype(np.float32)
				#source = np.expand_dims(source, -1)

				labels = self.labels[name]

				source = source.reshape((1,) + source.shape)

				yield source, labels


	def __len__ (self):
		return len(self.names)
