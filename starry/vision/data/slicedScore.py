
import os
import time
import numpy as np
import random
import cv2

from ..transform import Composer
from ..images import randomSliceImage, iterateSliceImage
from .utils import collateBatch, loadSplittedDatasets
from .cacheData import CachedIterableDataset
from .augmentor import Augmentor, appendGaussianNoise
from .score import makeReader, listAllScoreNames, STAFF



class SlicedScore (CachedIterableDataset):
	@staticmethod
	def load (root, args, splits, device='cpu'):
		return loadSplittedDatasets(SlicedScore, root=root, args=args, splits=splits, device=device)


	def __init__ (self, root, split='0/1', device='cpu', trans=[],
		labels=[], shuffle=False, slicing_width=256, crop_margin=0, augmentor=None,
		cache_batches=False):
		super().__init__(enable_cache=cache_batches and not shuffle)

		self.reader, self.root = makeReader(root)

		self.device = device
		self.labels = labels
		self.shuffle = shuffle
		self.slicing_width = slicing_width
		self.crop_margin = crop_margin
		self.batch_count = None

		self.names = listAllScoreNames(self.reader, split)
		self.trans = Composer(trans) if len(trans) > 0 else None

		self.tinter = None
		self.gaussian_noise = 0
		self.augmentor = Augmentor(augmentor, shuffle = self.shuffle)


	def augmentFeature (self, source):
		if self.tinter:
			source = self.tinter.tint(source)
		if self.gaussian_noise > 0:
			source = appendGaussianNoise(source, self.gaussian_noise)

		return np.clip(source, 0, 1)


	def collateBatchImpl (self, batch):
		return collateBatch(batch, self.trans, self.device, by_numpy=True)


	def _loadTarget (self, name, *_):
		imgs = [self.reader.readImage(os.path.join(label, name + ".png")) for label in self.labels]
		if None in imgs:
			return None

		return np.stack(
			[cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) for img in imgs],
			axis = -1)


	def iterImpl (self):
		if self.shuffle:
			random.shuffle(self.names)
			np.random.seed(int((time.time() * 1e+7 % 1e+7) + random.randint(0, 1e+5)))

		batches = 0

		for name in self.names:
			source_path = os.path.join(STAFF, name + ".png")
			if not self.reader.exists(source_path):
				self.names.remove(name)
				print('staff file missing, removed:', source_path)
				continue
			source = self.reader.readImage(source_path)

			# to gray
			source = cv2.cvtColor(source, cv2.COLOR_RGBA2GRAY)
			source = (source / 255.0).astype(np.float32)
			source = np.expand_dims(source, -1)

			target = self._loadTarget(name, source)
			if target is None:
				print(f'Target images loading of {name} failed.')
				continue

			if self.shuffle:
				batches += 1
				src, tar = randomSliceImage(source, target, self.slicing_width, crop_margin=self.crop_margin)
				src, tar = self.augmentor.augment(src, tar)

				yield src, tar
			else:
				for src, tar in iterateSliceImage(source, target, self.slicing_width, crop_margin=self.crop_margin):
					batches += 1

					np.random.seed(batches)
					src, tar = self.augmentor.augment(src, tar)
					yield src, tar

		self.batch_count = batches


	def __len__ (self):
		return self.batch_count if self.batch_count is not None else len(self.names) * (1 if self.shuffle else 6)
