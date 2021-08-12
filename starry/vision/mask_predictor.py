
import numpy as np
import torch
import logging
import PIL.Image

from ..utils.predictor import Predictor
from .images import arrayFromImageStream, sliceFeature, spliceOutputTensor, MARGIN_DIVIDER, maskToAlpha, encodeImageBase64
from . import transform



class StaffMask:
	def __init__ (self, hotmap):	# channel semantic: [fore, back]
		hotmap = maskToAlpha(hotmap, frac_y=True)
		self.image = PIL.Image.fromarray(hotmap, 'LA')

	def json (self):
		return {
			'image': encodeImageBase64(self.image),
		}


class MaskPredictor (Predictor):
	def __init__(self, config, device='cpu', **_):
		super().__init__(device=device)

		self.loadModel(config)

		self.slicing_width = config['data.slicing_width']
		self.labels = config['data.labels']

		trans = [t for t in config['data.trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)


	def predict (self, streams, **_):
		images = map(lambda stream: arrayFromImageStream(stream), streams)

		results = []
		for i, image in enumerate(images):
			#logging.info('feature: %s', image.shape)
			pieces = sliceFeature(image, width=self.slicing_width, overlapping=2 / MARGIN_DIVIDER, padding=False)
			pieces = np.array(list(pieces), dtype=np.float32)
			staves, _ = self.composer(pieces, np.ones((1, 4, 4, 2)))

			with torch.no_grad():
				batch = torch.from_numpy(staves)
				batch = batch.to(self.device)

				output = self.model(batch)	# (batch, channel, height, width)
				hotmap = spliceOutputTensor(output, soft = True)	# (channel, height, width)
				if hotmap.shape[2] > image.shape[1]:
					hotmap = hotmap[:, :, :image.shape[1]]
				#logging.info('hotmap: %s', hotmap.shape)

				results.append(StaffMask(hotmap).json())

		return results
