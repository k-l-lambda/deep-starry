
import os
import logging
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset
import cv2

from .utils import loadSplittedDatasets, parseFilterStr
from .imageReader import makeReader


def collateBatchSingle (batch, device):
	assert len(batch) == 1

	x, y = batch[0]

	return x.to(device), y.to(device)


class DimensionCluster:
	def __init__ (self, csv_file, cluster_size=0x100000, no_repeat=False, shuffle=False, filterStr=None):
		dataframes = pd.read_csv(csv_file)

		if filterStr is not None:
			phases, cycle = parseFilterStr(filterStr)
			dataframes = dataframes.filter(axis='index', items=[i for i in range(len(dataframes)) if (i % cycle) in phases])

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


	def __len__ (self):
		return len(self.name_dict.keys())


class SuperImage (IterableDataset):
	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None, **_):
		return loadSplittedDatasets(cls, root=root, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, split='0/1', dimensions=None, cluster_size=0x100000, downsample=4, device='cpu', shuffle=False, **_):
		self.reader, _ = makeReader(root)
		self.shuffle = shuffle
		self.device = device

		self.cluster = DimensionCluster(os.path.join(root, dimensions), cluster_size=cluster_size, no_repeat=not shuffle, shuffle=shuffle, filterStr=split)
		self.downsample = downsample


	def __iter__ (self):
		for (h, w), names in self.cluster:
			lh, lw = h // self.downsample, w // self.downsample

			x = np.zeros((len(names), 3, lh, lw))
			y = np.zeros((len(names), 3, h, w))

			for i, name in enumerate(names):
				if not self.reader.exists(name):
					logging.warn('image file missing: %s', name)
					continue

				image = self.reader.readImage(name)
				if len(image.shape) < 3:
					image = image.reshape(image.shape + (1,))
				if image.shape[2] == 1:
					image = np.concatenate((image, image, image), axis=2)
				elif image.shape[2] > 3:
					image = image[:, :, :3]

				lm, tm = random.randint(0, image.shape[1] - w), random.randint(0, image.shape[0] - h)
				image = image[tm:tm + h, lm:lm + w, :]

				y[i] = image.transpose(2, 0, 1) / 255.

				imageLow = cv2.resize(image, (image.shape[1] // self.downsample, image.shape[0] // self.downsample), interpolation=cv2.INTER_AREA)
				x[i] = imageLow.transpose(2, 0, 1) / 255.

			yield torch.from_numpy(x).float(), torch.from_numpy(y).float()


	def __len__ (self):
		return len(self.cluster)


	def collateBatch (self, batch):
		return collateBatchSingle(batch, self.device)
