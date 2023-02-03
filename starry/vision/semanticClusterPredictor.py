
import os
import numpy as np
import torch
import yaml
import logging

from ..utils.config import Configuration
from ..utils.predictor import Predictor
from .score_semantic import ScoreSemantic
from .images import arrayFromImageStream, sliceFeature, spliceOutputTensor, MARGIN_DIVIDER
from . import transform



class SemanticSubPredictor (Predictor):
	def __init__ (self, config, device='cpu'):
		super().__init__(device=device)

		self.loadModel(config)

		data_args = config['data.args']

		self.slicing_width = data_args['slicing_width']
		self.labels = data_args['labels']

		trans = [t for t in data_args['trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)


	def __call__ (self, feature, confidence_table):
		semantic = self.model(feature)
		semantic = spliceOutputTensor(semantic)

		return ScoreSemantic(np.uint8(semantic * 255), self.labels, confidence_table=confidence_table)


class SemanticClusterPredictor (Predictor):
	def __init__ (self, config, device='cpu', **_):
		super().__init__()

		self.confidence_table = None
		confidence_path = config.localPath('confidence.yaml')
		if confidence_path and os.path.exists(confidence_path):
			with open(confidence_path, 'r') as file:
				self.confidence_table = yaml.safe_load(file)
				logging.info('confidence_table loaded: %s', confidence_path)

		sub_configs = [Configuration(config.localPath(dirname)) for dirname in config['subs']]
		self.modules = [SemanticSubPredictor(cfg, device=device) for cfg in sub_configs]


	def predict (self, streams):
		for stream in streams:
			image = arrayFromImageStream(stream)

			pieces = sliceFeature(image, width=self.slicing_width, overlapping = 2 / MARGIN_DIVIDER, padding=True)
			pieces = np.array(list(pieces), dtype=np.uint8)
			staves, _ = self.composer(pieces, np.ones((1, 4, 4, 2)))

			with torch.no_grad():
				batch = torch.from_numpy(staves)
				batch = batch.to(self.device)

				ss = [module(batch, self.confidence_table) for module in self.modules]

				yield ScoreSemantic.merge(ss).json()
