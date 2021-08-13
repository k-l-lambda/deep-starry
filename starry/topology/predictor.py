
import torch
import logging

from .data import exampleToTensors, Dataset
from ..utils.predictor import Predictor



class TopologyPredictor (Predictor):
	def __init__(self, config, batch_size=4, device='cpu', **_):
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

		for matrix, mask in zip(matrices, masks):
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
					'matrixH': matrix,
					'mask': mask,
				}
