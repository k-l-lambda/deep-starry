
import numpy as np
import torch
import logging
import PIL.Image
import cv2

from ..utils.predictor import Predictor
from .images import arrayFromImageStream, encodeImageBase64, writeImageFileFormat
from . import transform



class PageLayout:
	def __init__ (self, hotmap):	# channel semantic: [VL, StaffBox, HL]
		lines_map = hotmap[2]
		self.interval = PageLayout.measureInterval(lines_map)

		hotmap = np.uint8(hotmap * 255)
		hotmap = np.moveaxis(hotmap, 0, -1)
		self.image = PIL.Image.fromarray(hotmap, 'RGB')

		staves_map = hotmap[:, :, 1]
		self.theta = PageLayout.measureTheta(staves_map)


	def json (self):
		image_code = encodeImageBase64(self.image)

		return {
			'image': image_code,
			'theta': self.theta,
			'interval': self.interval,
		}


	@staticmethod
	def measureTheta (hotmap):
		edges = cv2.Canny(hotmap, 50, 150, apertureSize = 3)
		lines = cv2.HoughLines(edges, 1, np.pi/18000, round(hotmap.shape[1] * 0.4), min_theta = np.pi * 0.48, max_theta = np.pi * 0.52)
		if lines is None:
			return None

		avg_theta = sum(map(lambda line: line[0][1], lines)) / len(lines)

		return avg_theta - np.pi / 2


	@staticmethod
	def measureInterval (hotmap):
		width = hotmap.shape[1]
		interval_min, interval_max = round(width * 0.004), round(width * 0.025)

		brights = []
		for y in range(interval_min, interval_max):
			m1, m2 = hotmap[y:], hotmap[:-y]
			p = np.multiply(m1, m2)
			brights.append(np.mean(p))

		return interval_min + int(np.argmax(brights))


class LayoutPredictor (Predictor):
	def __init__(self, config, device='cpu', inspect=False):
		super().__init__(device=device)

		self.inspect = inspect
		if inspect:
			config['model.type'] = config['model.type'] + 'Inspection'

		self.loadModel(config)

		trans = [t for t in config['data.trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)


	def predict (self, streams, output_path=None):
		images = map(lambda stream: arrayFromImageStream(stream), streams)

		for i, image in enumerate(images):
			image = np.expand_dims(image, 0)
			image = self.normalizeImageDimension(image)

			batch, _ = self.composer(image, np.ones((1, 4, 4, 2)))
			batch = torch.from_numpy(batch)
			batch = batch.to(self.device)

			with torch.no_grad():
				output = self.model(batch)
				output = output.cpu().numpy()
				hotmap = output[0]
				if self.inspect:
					hotmap = hotmap[0]	# (batch, channel, height, width)
				#logging.info('hotmaps: %s', hotmaps.shape)

				if output_path:
					page = np.clip(np.uint8(image[0] * 255), 0, 255)
					writeImageFileFormat(page, output_path, i, 'page')

					for i, layout in enumerate(output[1:]):
						layout = np.clip(np.uint8(layout * 255), 0, 255)
						layout = np.moveaxis(layout, 0, -1)[:, :, ::-1]
						writeImageFileFormat(layout, output_path, begin_index, f'layout-{i}')

				yield PageLayout(hotmap).json()


	@classmethod
	def normalizeImageDimension (cls, image):
		n, h, w, c = image.shape
		if h % 4 != 0 | w % 4 != 0:
			logging.warn('[LayoutPredictor]	image diemension cannot be divisible by 4: %d x %d', h, w)

			return image[:, :h - h % 4, :w - w % 4, :]

		return image
