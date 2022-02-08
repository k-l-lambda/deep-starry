
import os
import numpy as np
import random
import time
import cv2
import logging
import torch

from ..transform import Composer
from .utils import collateBatch, loadSplittedDatasets
from .score import makeReader, listAllScoreNames, PAGE, PAGE_LAYOUT
from .cacheData import CachedIterableDataset
from .augmentor import Augmentor



class ScorePage (CachedIterableDataset):
	@classmethod
	def load (cls, root, args, splits, device='cpu', args_variant=None):
		return loadSplittedDatasets(cls, root=root, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, split='0/1', device='cpu', trans=[], shuffle=False, augmentor=None, cache_batches=False, **kwargs):
		super().__init__(enable_cache=cache_batches and not shuffle)

		self.reader, self.root = makeReader(root)

		self.device = device
		self.shuffle = shuffle

		self.names = listAllScoreNames(self.reader, split, dir=PAGE)
		self.trans = Composer(trans) if len(trans) > 0 else None

		self.gaussian_noise = 0
		self.augmentor = augmentor and Augmentor(augmentor, shuffle = self.shuffle)
		self.channel_order = augmentor.get('channel_order') if augmentor else None


	def collateBatchImpl (self, batch):
		return collateBatch(batch, self.trans, self.device, by_numpy=True)


	def loadTarget (self, name):
		layout = self.reader.readImage(os.path.join(PAGE_LAYOUT, name + ".png"))
		#print('mask:', mask.shape)

		if layout is None:
			return None

		result = layout[:, :, :3]

		if self.channel_order is not None:
			channels = [result[:, :, c] for c in self.channel_order]
			result = np.stack(channels, axis = 2)

		return result


	def iterImpl (self):
		if self.shuffle:
			random.shuffle(self.names)
			np.random.seed(int((time.time() * 1e+7 % 1e+7) + random.randint(0, 1e+5)))

		for i, name in enumerate(self.names):
			source_path = os.path.join(PAGE, name + ".png")
			if not self.reader.exists(source_path):
				self.names.remove(name)
				logging.warn('staff file missing, removed: %s', source_path)
				continue
			source = self.reader.readImage(source_path)

			# to gray
			source = cv2.cvtColor(source, cv2.COLOR_RGBA2GRAY)
			source = (source / 255.0).astype(np.float32)
			source = np.expand_dims(source, -1)

			target = self.loadTarget(name)
			if target is None:
				logging.warn(f'Target images loading of {name} failed.')
				continue

			if not self.shuffle:
				np.random.seed(i)
			source, target = self.augmentor.augment(source, target)

			yield source, target


	def __len__ (self):
		return len(self.names)


class ScorePageRaw (ScorePage):
	def __init__ (self, root, **kwargs):
		super().__init__(root, **kwargs)
		logging.info('ScorePageRaw.__init__')


	def collateBatch (self, batch):
		name, image, label = batch[0]
		images = np.stack([image], axis=0)
		labels = np.stack([label], axis=0)

		if self.trans is not None:
			images, labels = self.trans(images, labels)
		feature = torch.from_numpy(images).to(self.device)
		target = torch.from_numpy(labels).to(self.device)

		return name, feature, target


	def iterImpl (self):
		for i, name in enumerate(self.names):
			source_path = os.path.join(PAGE, name + ".png")
			if not self.reader.exists(source_path):
				self.names.remove(name)
				logging.warn('staff file missing, removed: %s', source_path)
				continue
			source = self.reader.readImage(source_path)

			# to gray
			source = cv2.cvtColor(source, cv2.COLOR_RGBA2GRAY)
			source = (source / 255.0).astype(np.float32)
			source = np.expand_dims(source, -1)

			target = self.loadTarget(name)

			yield name, source, target
