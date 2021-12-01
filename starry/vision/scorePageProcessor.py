
import os
import numpy as np
import torch
import logging
import PIL.Image
import cv2

from ..utils.predictor import Predictor
from .layout_predictor import PageLayout
from . import transform



BATCH_SIZE = int(os.environ.get('SCORE_PAGE_PROCESSOR_BATCH_SIZE', '2'))

RESIZE_WIDTH = 600


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
		for i in range(0, len(input_paths), BATCH_SIZE):
			images = list(map(lambda path: cv2.imread(path), input_paths[i:i + BATCH_SIZE]))

			# unify images' dimensions
			ratio = min(map(lambda img: img.shape[0] / img.shape[1], images))
			height = int(RESIZE_WIDTH * ratio)
			height -= height % 4
			unified_images = list(map(lambda img: cv2.resize(img, (RESIZE_WIDTH, img.shape[0] * RESIZE_WIDTH // img.shape[1]))[:height], images))
			image_array = np.stack(unified_images, axis=0)

			batch, _ = self.composer(image_array, np.ones((1, 4, 4, 2)))
			batch = torch.from_numpy(batch)
			batch = batch.to(self.device)

			with torch.no_grad():
				output = self.model(batch)
				output = output.cpu().numpy()

				for j, heatmap in enumerate(output):
					layout = PageLayout(heatmap)
					yield {'theta': layout.theta, 'interval': layout.interval}
