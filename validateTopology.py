
import sys
import os
import argparse
import logging
import torch

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.utils.predictor import Predictor
from starry.topology.viewer import DatasetViewer



# workaround cuda unavailable issue
torch.cuda.is_available()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')


class Validator (Predictor):
	def __init__ (self, config, device, n_axes=4):
		super().__init__(device=device)

		self.viewer = DatasetViewer(config, n_axes=n_axes)

		self.loadModel(config)


	def run (self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			pred = self.model(tensors)
			#print('pred:', [*pred[0].keys()])
			self.viewer.showEventTopology(tensors, pred)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-d', '--data', type=str, help='data configuration file')
	parser.add_argument('-s', '--splits', type=str, default='0/10')
	parser.add_argument('-ax', '--n_axes', type=int, default=4)
	parser.add_argument('-dv', '--device', type=str, default='cuda')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)

	if args.data:
		data_config = Configuration.createOrLoad(args.data, volatile=True)
		config['data'] = data_config['data']

	if args.splits is not None:
		config['data.splits'] = args.splits

	data, = loadDataset(config, data_dir=DATA_DIR, device=args.device)
	validator = Validator(config, n_axes=args.n_axes, device=args.device)

	validator.run(data)


if __name__ == '__main__':
	main()
