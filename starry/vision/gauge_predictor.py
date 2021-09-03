
import numpy as np
import torch
import logging
import PIL.Image

from ..utils.predictor import Predictor
from .images import arrayFromImageStream, sliceFeature, spliceOutputTensor, MARGIN_DIVIDER, gaugeToRGB, encodeImageBase64
from . import transform



class StaffGauge:
	def __init__ (self, hotmap):	# channel semantic: [fore, back]
		hotmap = gaugeToRGB(hotmap, frac_y=True)
		self.image = PIL.Image.fromarray(hotmap[:, :, ::-1], 'RGB')

	def json (self):
		return {
			'image': encodeImageBase64(self.image),
		}


class GaugePredictor (Predictor):
	def __init__(self, config, device='cpu', **_):
		super().__init__(device=device)

		self.loadModel(config)

		data_args = config['data.args'] or config['data']

		self.slicing_width = data_args['slicing_width']

		trans = [t for t in data_args['trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)


	def predict (self, streams, **_):
		images = map(lambda stream: arrayFromImageStream(stream), streams)

		for i, image in enumerate(images):
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

				yield StaffGauge(hotmap).json()
