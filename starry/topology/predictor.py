
import os
import torch
import logging

from .data import exampleToTensors, Dataset
from .semantic_element import SemanticElementType
from ..utils.predictor import Predictor



BATCH_SIZE = int(os.environ.get('TOPOLOGY_PREDICTOR_BATCH_SIZE', '1'))


class TopologyPredictorH (Predictor):
	def __init__(self, config, batch_size=BATCH_SIZE, device='cpu', **_):
		super().__init__(batch_size=batch_size, device=device)

		self.d_model = config['model.args.d_model']

		self.loadModel(config)


	def predict(self, clusters, expand=False):
		n_seq_max = max(len(cluster['elements']) for cluster in clusters)
		examples = list(map(lambda ex: exampleToTensors(ex, n_seq_max, self.d_model, matrix_placeholder=True), clusters))
		dataset = Dataset(examples, self.batch_size, self.device)

		matrices = []
		masks = []
		with torch.no_grad():
			for batch in dataset:
				matrices += self.model(batch['seq_id'], batch['seq_position'], batch['mask'])
				masks += batch['mask']

		for matrix, mask, cluster in zip(matrices, masks, clusters):
			matrix = matrix.cpu().tolist()
			mask = mask.cpu().tolist()

			if expand:
				it = iter(matrix)
				full = [
					[
						next(it) if row & column else None
						for column in mask[1]
					] for row in mask[0]
				]
				yield full
			else:
				yield {
					'index': cluster.get('index'),
					'matrixH': matrix,
					'mask': mask,
				}


class TopologyPredictorHV (Predictor):
	def __init__(self, config, batch_size=BATCH_SIZE, device='cpu', **_):
		super().__init__(batch_size=batch_size, device=device)

		self.d_model = config['model.args.d_model']

		self.loadModel(config)


	def predict(self, clusters):
		# filter out new elements out of SemanticElementType.MAX
		for cluster in clusters:
			cluster['elements'] = list(filter(lambda e: e['type'] < SemanticElementType.MAX, cluster['elements']))

		n_seq_max = max(len(cluster['elements']) for cluster in clusters)
		examples = list(map(lambda ex: exampleToTensors(ex, n_seq_max, self.d_model, matrix_placeholder=True, pruned_maskv=True), clusters))
		dataset = Dataset(examples, self.batch_size, self.device)

		matricesH = []
		matricesV = []
		masks = []
		with torch.no_grad():
			for batch in dataset:
				h, v = self.model(batch['seq_id'], batch['seq_position'], batch['mask'])
				matricesH += h
				matricesV += v

				masks += batch['mask']

		for matrixH, matrixV, mask, cluster in zip(matricesH, matricesV, masks, clusters):
			matrixH = matrixH.cpu().tolist()
			matrixV = matrixV.cpu().tolist()
			mask = mask.cpu().tolist()

			yield {
				'index': cluster.get('index'),
				'matrixH': matrixH,
				'matrixV': matrixV,
				'mask': mask,
			}
