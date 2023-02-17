
import sys
import os
import numpy as np
import random
import time
import cv2
import json
import re
import logging
import yaml

from ..transform import Composer
from ..images import randomSliceImage, iterateSliceImage
from .utils import collateBatch, loadSplittedDatasets
from .cacheData import CachedIterableDataset
from .imageReader import CachedImageReader, _S
from .augmentor import Augmentor
from .score import makeReader, listAllScoreNames, GRAPH, MASK, STAFF



SemanticGroups = yaml.safe_load(open('assets/semanticGroups.yaml', 'r'))['Glyph']


def normalizeName (name):
	return name.split('.')[-1]


# target: (height, width, channel)
def renderTargetFromGraph (graph, labels, size, unit_size=16, point_radius=2 / 16,
	line_thickness=2 / 16, blur_scale=64, reader=None, name=None):
	#print('graph:', graph)
	upscale = 4
	upsize = (size[0] * upscale, size[1] * upscale)
	upunit_size = unit_size * upscale
	y0 = upsize[0] // 2

	transX = lambda x: min(max(round(x * upunit_size), 0), upsize[1])
	transY = lambda y: min(max(round(y * upunit_size) + y0, 0), upsize[0])

	layers = []
	for i, label in enumerate(labels):
		#cache_path = os.path.join(cache_dir, label, name + ".png") if cache_dir else None
		#layer = cv2.imread(cache_path) if cache_path else None
		file_path = os.path.join(label, normalizeName(name) + ".png")
		layer = reader.readImage(file_path) if reader and reader.exists(file_path) else None
		assert layer is None or layer.shape == size, f'layer shape mismatch: {layer.shape}, {size}'

		is_point_of_label = lambda point: (point['semantic'] in SemanticGroups[label]) if SemanticGroups.get(label) else (point['semantic'] == label)

		if layer is None:
			points = filter(is_point_of_label, graph['points'])
			layer = np.zeros(upsize, dtype=np.uint8)

			if re.match(r'^vline_', label):
				for point in points:
					x = round(point['x'] * upunit_size)
					if x >= 0 and x < upsize[1]:
						y1 = transY(point['extension']['y1'])
						y2 = transY(point['extension']['y2'])
						cv2.line(layer, (x, y1), (x, y2), 255, round(line_thickness * upunit_size))
			elif re.match(r'^rect_', label):
				for point in points:
					x1 = transX(point['x'] - point['extension']['width'] / 2)
					x2 = transX(point['x'] + point['extension']['width'] / 2)
					y1 = transY(point['y'] - point['extension']['height'] / 2)
					y2 = transY(point['y'] + point['extension']['height'] / 2)
					cv2.rectangle(layer, (x1, y1), (x2, y2), 255, -1)
			else:
				for point in points:
					cx = round(point['x'] * upunit_size)
					cy = round(point['y'] * upunit_size + y0)
					if cx >= 0 and cx < upsize[1] and cy >= 0 and cy < upsize[0]:
						cv2.circle(layer, (cx, cy), round(point_radius * upunit_size), 255, -1)

			blur_kernel = (upsize[0] // (blur_scale * 2)) * 2 + 1
			if blur_kernel > 1:
				layer = cv2.GaussianBlur(layer, (blur_kernel, blur_kernel), 0)
			layer = cv2.resize(layer, size[::-1])	# 'cv2.resize' is weird, layer shape is size now, NOT size[::-1]

			if reader:
				#print('cache_path:', cache_path, cache_dir)
				reader.writeImage(file_path, layer)
		#else:
		#	print('layer loaded:', name, label, layer.shape)

		layers.append(layer)

	return np.stack(layers, axis=-1)


class RenderScore (CachedIterableDataset):
	@staticmethod
	def load (root, args, splits, device='cpu', args_variant=None):
		return loadSplittedDatasets(RenderScore, root=root, args=args, splits=splits, device=device, args_variant=args_variant)


	def __init__ (self, root, split='0/1', device='cpu', trans=[],
		labels=[], shuffle=False, slicing_width=256, blur_scale=None,
		crop_margin=0, input_mask=False, unit_size=8, cache_labels=False, augmentor=None,
		cache_batches=False,
	):
		super().__init__(enable_cache=cache_batches and not shuffle)

		self.reader, self.root = makeReader(root)

		self.device = device
		self.labels = labels
		self.shuffle = shuffle
		self.slicing_width = slicing_width
		self.blur_scale = 64 if blur_scale is None else blur_scale
		self.crop_margin = crop_margin
		self.input_mask = input_mask
		self.unit_size = unit_size
		self.cache_dir = os.path.join(self.root, ".cache") if cache_labels else None
		self.batch_count = None
		self.chosen_channels = list(range(len(labels)))

		if self.cache_dir:
			os.makedirs(self.cache_dir, exist_ok = True)
			for label in self.labels:
				dir = os.path.join(self.cache_dir, label)
				os.makedirs(dir, exist_ok = True)

			self.cachedReader = CachedImageReader(self.cache_dir)

		self.names = listAllScoreNames(self.reader, split)
		self.trans = Composer(trans) if len(trans) > 0 else None

		self.augmentor = Augmentor(augmentor, shuffle = self.shuffle)

		if len(self.names) == 0:
			logging.warn('[RenderScore]	dataset is empty for split "%s"', split)


	def collateBatchImpl (self, batch):
		return collateBatch(batch, self.trans, self.device, by_numpy=True)


	def iterImpl (self):
		if self.shuffle:
			random.shuffle(self.names)
			np.random.seed(int((time.time() * 1e+7 % 1e+7) + random.randint(0, 1e+5)))

		batches = 0

		for name in self.names:
			#print('name:', name)
			source_path = os.path.join(STAFF, name + ".png")
			if not self.reader.exists(source_path):
				self.names.remove(name)
				logging.warn('staff file missing, removed: %s', source_path)
				continue
			source = self.reader.readImage(source_path)

			# to gray
			source = cv2.cvtColor(source, cv2.COLOR_RGBA2GRAY)
			source = (source / 255.0).astype(np.float32)
			source = np.expand_dims(source, -1)

			if self.input_mask:
				mask_path = os.path.join(MASK, name + ".png")
				if not self.reader.exists(mask_path):
					logging.warn('mask file missing: %s', mask_path)
					continue
				mask = self.reader.readImage(mask_path)[:, :, :2]
				source[:, :, 1:] = mask[:, :source.shape[1], ::-1]

			graph_path = _S(os.path.join(GRAPH, name + ".json"))
			if not self.reader.exists(graph_path):
				logging.warn('graph file missing: %s', graph_path)
				continue
			graph = None
			with self.reader.fs.open(graph_path, 'rt', encoding='UTF-8') as graph_file:
				try:
					graph = json.load(graph_file, strict=False)
				except:
					logging.warn('error to load graph: %s, %s', graph_path, sys.exc_info()[1])
					continue

			labels = [self.labels[c] for c in self.chosen_channels]

			target = renderTargetFromGraph(graph, labels, source.shape[:2], blur_scale=self.blur_scale, unit_size=self.unit_size,
				reader=self.cachedReader, name=name)

			if self.shuffle:
				batches += 1

				src, tar = randomSliceImage(source, target, self.slicing_width)
				src, tar = self.augmentor.augment(src, tar)

				yield src, tar
			else:
				for src, tar in iterateSliceImage(source, target, self.slicing_width, crop_margin = self.crop_margin):
					batches += 1

					np.random.seed(batches)
					src, tar = self.augmentor.augment(src, tar)
					yield src, tar

		self.batch_count = batches


	def __len__ (self):
		if self.batch_count is not None:
			return self.batch_count

		return len(self.names) if self.shuffle else len(self.names) * 10
