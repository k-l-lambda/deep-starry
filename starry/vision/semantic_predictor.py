
import os
import numpy as np
import torch
import yaml
import logging

from ..utils.predictor import Predictor
from .images import arrayFromImageStream, writeImageFileFormat, sliceFeature, spliceOutputTensor, MARGIN_DIVIDER
from . import transform
from .chromaticChannels import composeChromaticMap
from .score_semantic import ScoreSemantic



class SemanticPredictor(Predictor):
	def __init__(self, config, device='cpu', inspect=False):
		super().__init__(device=device)

		self.inspect = inspect
		if inspect:
			config['model.type'] = config['model.type'] + 'Inspection'

		self.loadModel(config)

		self.slicing_width = config['data.slicing_width']
		self.labels = config['data.labels']

		trans = [t for t in config['data.trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)

		self.confidence_table = None
		confidence_path = config.localPath('confidence.yaml')
		if os.path.exists(confidence_path):
			with open(confidence_path, 'r') as file:
				self.confidence_table = yaml.safe_load(file)
				logging.info('confidence_table loaded: %s', confidence_path)


	def predict (self, streams, output_path=None):
		images = map(lambda stream: arrayFromImageStream(stream), streams)

		graphs = []
		for i, image in enumerate(images):
			if output_path:
				writeImageFileFormat(image, output_path, i, 'feature')

			pieces = sliceFeature(image, width=self.slicing_width, overlapping = 2 / MARGIN_DIVIDER, padding=True)
			pieces = np.array(list(pieces), dtype=np.uint8)
			staves, _ = self.composer(pieces, np.ones((1, 4, 4, 2)))

			with torch.no_grad():
				batch = torch.from_numpy(staves)
				batch = batch.to(self.device)

				output = self.model(batch)

				semantic, mask = None, None
				if self.inspect:
					mask, semantic = output
				else:
					semantic = output	# (batch, channel, height, width)
				semantic, mask = map(spliceOutputTensor, (semantic, mask))

				if output_path:
					if mask is not None:
						mask = np.concatenate([np.zeros((1, mask.shape[1], mask.shape[2])), mask], axis = 0)
						mask = np.moveaxis(mask, 0, -1)
						mask = np.clip(np.uint8(mask * 255), 0, 255)

						writeImageFileFormat(mask, output_path, i, 'mask')

					chromatic = composeChromaticMap(semantic)
					writeImageFileFormat(chromatic, output_path, i, 'semantics')

				ss = ScoreSemantic(np.uint8(semantic * 255), self.labels, confidence_table=self.confidence_table)
				graphs.append(ss.json())

		return graphs
