
import sys
import os
import argparse
import logging
import torch

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.utils.predictor import Predictor
from starry.paraff.viewer import ParaffViewer



# workaround cuda unavailable issue
torch.cuda.is_available()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')


class Validator (Predictor):
	def __init__ (self, config, device='cuda', **kw_args):
		super().__init__(device=device)

		self.viewer = ParaffViewer(config, **kw_args)

		self.loadModel(config, postfix='Loss')


	def run (self, data):
		for i, batch in enumerate(data):
			logging.info('batch: %d', i)

			with torch.no_grad():
				inspection = self.model.inspectRun(batch)

			self.viewer.show(batch, inspection)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-d', '--data', type=str, help='data configuration file')
	parser.add_argument('-s', '--splits', type=str, default='0/1')
	parser.add_argument('-dv', '--device', type=str, default='cpu')
	parser.add_argument('-z', '--show_latent', action='store_true')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)

	if args.data:
		data_config = Configuration.createOrLoad(args.data, volatile=True)
		config['data'] = data_config['data']

	if args.splits is not None:
		config['data.splits'] = args.splits

	config['data.batch_size'] = 1

	data, = loadDataset(config, data_dir=DATA_DIR, device=args.device)
	validator = Validator(config, device=args.device, show_latent=args.show_latent)

	validator.run(data)


if __name__ == '__main__':
	main()
