
import torch
import logging

from .models import TransformJointerLoss
from .data import exampleToTensors, Dataset



class Predictor:
	def __init__(self, config, batch_size=4, device='cpu', **_):
		self.batch_size = batch_size
		self.device = device

		model = TransformJointerLoss(**config['model.args'])
		if config['best']:
			checkpoint = torch.load(config.localPath(config['best']), map_location=self.device)
			model.load_state_dict(checkpoint['model'])
			logging.info(f'checkpoint loaded: {config["best"]}')

		self.d_model = config['model.args.d_model']

		self.model = model.deducer
		self.model.to(self.device)
		self.model.eval()


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

		results = []
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
				results.append(full)
			else:
				results.append({
					'matrixH': matrix,
					'mask': mask,
				})

		return results
