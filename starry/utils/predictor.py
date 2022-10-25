
import os
import torch
import logging

from .model_factory import loadModel
from .check_host import check_host



class Predictor:
	def __init__(self, batch_size=1, device='cpu'):
		self.batch_size = batch_size
		self.device = device


	def loadModel (self, config):
		weights_filename = config['best']
		if weights_filename and os.path.exists(config.localPath(weights_filename + '.pt')):
			self.model = torch.jit.load(config.localPath(weights_filename + '.pt'))
			logging.info(f'checkpoint loaded: {weights_filename}.pt')
		else:
			self.model = loadModel(config['model'])
			if weights_filename:
				if not os.path.exists(config.localPath(weights_filename)):
					weights_filename += '.chkpt'

				checkpoint = torch.load(config.localPath(weights_filename), map_location=self.device)
				self.model.load_state_dict(checkpoint['model'])
				logging.info(f'checkpoint loaded: {weights_filename}')

		self.model.to(self.device)
		self.model.eval()


	def checkHost (self):
		return check_host()
