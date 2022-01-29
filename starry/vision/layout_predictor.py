
import numpy as np
import torch
import logging
#import PIL.Image
import cv2

from ..utils.predictor import Predictor
from .images import arrayFromImageStream, writeImageFileFormat
from .scorePageLayout import PageLayout, RESIZE_WIDTH
from . import transform



def resizePageImage (img, size):
	w, h = size
	filled_height = img.shape[0] * w // img.shape[1]
	img = cv2.resize(img, (w, filled_height), interpolation=cv2.INTER_AREA)

	if filled_height < h:
		result = np.ones((h, w, img.shape[2]), dtype=np.uint8) * 255
		result[:filled_height] = img

		return result

	return img[:h]


class LayoutPredictor (Predictor):
	def __init__(self, config, device='cpu', inspect=False):
		super().__init__(device=device)

		self.inspect = inspect
		if inspect:
			config['model.type'] = config['model.type'] + 'Inspection'

		self.loadModel(config)

		data_args = config['data.args'] or config['data']

		trans = [t for t in data_args['trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)


	def predict (self, streams, advanced=False, **kwargs):
		if not advanced:
			for x in self.predictBasic(streams, **kwargs):
				yield x
		else:
			for x in self.predictDetection(streams):
				yield x


	def predictBasic (self, streams, output_path=None):
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
						writeImageFileFormat(layout, output_path, i, 'layout')

				yield PageLayout(hotmap).json()


	def predictDetection (self, streams):
		images = [arrayFromImageStream(stream) for stream in streams]

		ratio = max(img.shape[0] / img.shape[1] for img in images)
		height = int(RESIZE_WIDTH * ratio)
		height += -height % 4
		unified_images = [resizePageImage(img, (RESIZE_WIDTH, height)) for img in images]
		image_array = np.stack(unified_images, axis=0)

		batch, _ = self.composer(image_array, np.ones((1, 4, 4, 2)))
		batch = torch.from_numpy(batch)
		batch = batch.to(self.device)

		with torch.no_grad():
			output = self.model(batch)
			output = output.cpu().numpy()

			for i, heatmap in enumerate(output):
				image = images[i]

				layout = PageLayout(heatmap)
				result = layout.detect(image, ratio)

				if result and self.inspect:
					result['image'] = layout.json()['image']

				yield result


	@classmethod
	def normalizeImageDimension (cls, image):
		n, h, w, c = image.shape
		if (h % 4 != 0) | (w % 4 != 0):
			logging.warn('[LayoutPredictor]	image diemension cannot be divisible by 4: %d x %d', h, w)

			return image[:, :h - h % 4, :w - w % 4, :]

		return image
