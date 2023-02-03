
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

		self.labels = config['data.args.labels']


	def __call__ (self, feature, confidence_table):
		semantic = self.model(feature)
		semantic = spliceOutputTensor(semantic)

		return ScoreSemantic(np.uint8(semantic * 255), self.labels, confidence_table=confidence_table)


class SemanticClusterPredictor (Predictor):
	def __init__ (self, config, device='cpu', **_):
		super().__init__()

		self.slicing_width = config['predictor.slicing_width']
		self.confidence_table = config['predictor.confidence_table']

		sub_configs = [Configuration(config.localPath(dirname)) for dirname in config['subs']]
		self.modules = [SemanticSubPredictor(cfg, device=device) for cfg in sub_configs]

		trans = config['predictor.trans']
		self.composer = transform.Composer(trans)


	def predict (self, streams, **_):
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
