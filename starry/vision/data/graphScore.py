
import sys
import os
import torch
import math
import numpy as np
import random
import time
import cv2
import json
import re

from ..transform import Composer
from ..images import randomSliceImage, iterateSliceImage
from .cacheData import CachedIterableDataset
from .imageReader import ImageReader, CachedImageReader, _S
from .augmentor import Augmentor
from .score import makeReader, listAllScoreNames, STAFF, GRAPH



def sliceToBatch (source, width, overlapping=0.25, crop_margin=0):
	tw = width
	step = math.floor(width - overlapping * source.shape[0])

	pieces = []
	for x in range(crop_margin, source.shape[1] - crop_margin, step):
		sliced_source = np.ones((source.shape[0], width, source.shape[2]), dtype=np.float32)

		tx = x
		if x + width <= source.shape[1]:
			sliced_source = source[:, x:x + width, :]
		else:
			# fill zeros for right residue
			sliced_source[:, :source.shape[1] - x, :] = source[:, x:, :]

		pieces.append(sliced_source)

	return np.stack(pieces, axis=0)


class GraphScore:
	def __init__ (self, root, split='0/1', device='cpu', trans=[],
		shuffle=False, slicing_width=256, unit_size=8, cache_labels=False, augmentor=None,
		**_,
	):
		self.reader, self.root = makeReader(root)

		self.device = device
		self.shuffle = shuffle
		self.slicing_width = slicing_width
		#self.unit_size = unit_size

		self.names = listAllScoreNames(self.reader, split)
		self.trans = Composer(trans) if len(trans) > 0 else None

		self.augmentor = Augmentor(augmentor)


	def __iter__ (self):
		if self.shuffle:
			random.shuffle(self.names)
			np.random.seed(int((time.time() * 1e+7 % 1e+7) + random.randint(0, 1e+5)))

		for name in self.names:
			#print('name:', name)
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

			graph_path = _S(os.path.join(GRAPH, name + ".json"))
			if not self.reader.exists(graph_path):
				print('graph file missing:', graph_path)
				continue
			graph = None
			with self.reader.fs.open(graph_path, 'rt', encoding='UTF-8') as graph_file:
				try:
					graph = json.load(graph_file, strict=False)
				except:
					print('error to load graph:', graph_path, sys.exc_info()[1])
					continue

			source, _ = self.augmentor.augment(source)
			source = sliceToBatch(source, self.slicing_width)
			source, _ = self.trans(source, np.ones((1, 4, 4, 2)))
			source = torch.from_numpy(source).to(self.device)

			yield name, source, graph


	def __len__ (self):
		return len(self.names)
