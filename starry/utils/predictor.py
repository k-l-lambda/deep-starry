
import torch
import logging

from .model_factory import loadModel



class Predictor:
	def __init__(self, batch_size=1, device='cpu'):
		self.batch_size = batch_size
		self.device = device


	def loadModel (self, config):
		self.model = loadModel(config['model'])
		if config['best']:
			checkpoint = torch.load(config.localPath(config['best']), map_location=self.device)
			self.model.load_state_dict(checkpoint['model'])
			logging.info(f'checkpoint loaded: {config["best"]}')

		self.model.to(self.device)
		self.model.eval()
