
import torch

from .models import TransformJointerLoss
from .data import exampleToTensors, Dataset



class Predictor:
	def __init__(self, config, batch_size=4, device='cpu'):
		self.batch_size = batch_size
		self.device = device

		model = TransformJointerLoss(**config['model.args'])
		if config['best']:
			checkpoint = torch.load(config.localPath(config['best']), map_location=self.device)
			model.load_state_dict(checkpoint['model'])

		self.d_model = config['model.args.d_model']

		self.model = model.deducer
		self.model.to(self.device)
		self.model.eval()


	def predict(self, clusters):
		n_seq_max = max(len(cluster['elements']) for cluster in clusters)
		examples = list(map(lambda ex: exampleToTensors(ex, n_seq_max, self.d_model, matrix_placeholder=True), clusters))
		dataset = Dataset(examples, self.batch_size, self.device)

		results = []
		with torch.no_grad():
			for batch in dataset:
				results += self.model(batch['seq_id'], batch['seq_position'], batch['mask'])

		return results
