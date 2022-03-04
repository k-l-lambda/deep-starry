
import os
import torch
import logging

from ..utils.predictor import Predictor
from .data.events import clusterFeatureToTensors



BATCH_SIZE = int(os.environ.get('TOPOLOGY_PREDICTOR_BATCH_SIZE', '1'))


class EvtopoPredictor (Predictor):
	def __init__(self, config, batch_size=BATCH_SIZE, device='cpu', **_):
		super().__init__(batch_size=batch_size, device=device)

		self.loadModel(config)


	def predict(self, clusters):
		for i in range(0, len(clusters), self.batch_size):
			batch_clusters = clusters[i:i+self.batch_size]
			inputs = clusterFeatureToTensors(batch_clusters)

			with torch.no_grad():
				rec, mat = self.model(inputs)

			for ii, cluster in enumerate(batch_clusters):
				result = {
					'index': cluster.get('index'),
					'matrixH': mat[ii].cpu().tolist(),
				}

				for k, tensor in rec.items():
					result[k] = tensor[ii].cpu().tolist()

				yield result
