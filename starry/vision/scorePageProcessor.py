
import os
import sys
import io
import numpy as np
import torch
import logging
import PIL.Image
import cv2
import hashlib
import shutil

from ..utils.predictor import Predictor
from .scorePageLayout import PageLayout, RESIZE_WIDTH
from . import transform



BATCH_SIZE = int(os.environ.get('SCORE_PAGE_PROCESSOR_BATCH_SIZE', '1'))


def loadImageWithHash (path):
	bytes = open(path, 'rb').read()
	hash = hashlib.md5(bytes).hexdigest()
	image = PIL.Image.open(io.BytesIO(bytes)).convert('RGBA')
	arr = np.array(image)
	if len(arr.shape) > 2 and arr.shape[2] == 4:
		alpha = arr[:, :, 3]
		rgb = arr[:, :, :3]
		rgb[alpha == 0] = 255

		arr = rgb

	return arr, hash


def resizePageImage (img, size):
	w, h = size
	filled_height = img.shape[0] * w // img.shape[1]
	img = cv2.resize(img, (w, filled_height), interpolation=cv2.INTER_AREA)

	if filled_height < h:
		result = np.ones((h, w, img.shape[2]), dtype=np.uint8) * 255
		result[:filled_height] = img

		return result

	return img[:h]


class ScorePageProcessor (Predictor):
	def __init__(self, config, device='cpu', inspect=False):
		super().__init__(device=device)

		self.inspect = inspect
		if inspect:
			config['model.type'] = config['model.type'] + 'Inspection'

		self.loadModel(config)

		data_args = config['data.args'] or config['data']

		trans = [t for t in data_args['trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)


	def predict (self, input_paths, output_folder=None):
		if output_folder is not None:
			os.makedirs(output_folder, exist_ok=True)

		try:
			for i in range(0, len(input_paths), BATCH_SIZE):
				image_hashes = list(map(loadImageWithHash, input_paths[i:i + BATCH_SIZE]))
				page_filenames = None
				if output_folder:
					page_filenames = []
					for ii, ih in enumerate(image_hashes):
						input_path = input_paths[i + ii]
						_, hash = ih
						# get extension name from input path
						ext = os.path.splitext(input_path)[1]
						filename = hash + ext
						page_filenames.append(filename)
						shutil.copyfile(input_path, os.path.join(output_folder, filename))

						logging.debug('Page image copied: %s', filename)

				images = list(map(lambda ih: ih[0], image_hashes))
				#cv2.imwrite('./output/image0.png', images[0])

				for j, result in enumerate(self.predictImages(images, output_folder=output_folder)):
					if result['page_info'] is not None:
						result['page_info']['url'] = 'md5:' + page_filenames[j] if page_filenames is not None else None
						result['page_info']['path'] = input_paths[i + j]

					yield result

		except:
			logging.warn(sys.exc_info()[1])
			yield None


	def predictImages (self, images, output_folder=None):
		# unify images' dimensions
		ratio = max(map(lambda img: img.shape[0] / img.shape[1], images))
		height = int(RESIZE_WIDTH * ratio)
		height += -height % 4
		unified_images = list(map(lambda img: resizePageImage(img, (RESIZE_WIDTH, height)), images))
		image_array = np.stack(unified_images, axis=0)

		batch, _ = self.composer(image_array, np.ones((1, 4, 4, 2)))
		batch = torch.from_numpy(batch)
		batch = batch.to(self.device)

		with torch.no_grad():
			output = self.model(batch)
			output = output.cpu().numpy()

			for j, heatmap in enumerate(output):
				try:
					layout = PageLayout(heatmap)

					if layout.theta is None:
						yield {
							'theta': None,
							'interval': None,
							'detection': None,
							'page_info': None,
						}
						continue

					image = images[j]
					original_size = (image.shape[1], image.shape[0])

					result = layout.detect(image, ratio, output_folder=output_folder)
					result['page_info'] = {
						'size': original_size,
					}

					yield result
				except:
					logging.warn(sys.exc_info()[1])
					yield {
						'theta': None,
						'interval': None,
						'detection': None,
						'page_info': None,
						'reason': str(sys.exc_info()[1]),
					}
