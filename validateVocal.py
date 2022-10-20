
import sys
import os
import argparse
import logging
import torch

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.utils.predictor import Predictor
from starry.melody.vocalViewer import VocalViewer



# workaround cuda unavailable issue
torch.cuda.is_available()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')


class Validator (Predictor):
	def __init__ (self, config, args):
		super().__init__(device=args.device)

		self.viewer = VocalViewer(config, n_axes=args.n_axes, detail_mode=args.vocal_detail)

		self.loadModel(config)


	def run (self, data):
		for batch, tensors in enumerate(data):
			logging.info('batch: %d', batch)

			pitch, gain, mp, mt = tensors['pitch'], tensors['gain'], tensors['midi_pitch'], tensors['midi_rtick']

			with torch.no_grad():
				pred = self.model(pitch, gain, mp, mt)
				#print('pred:', pred)

			self.viewer.showVocalBatch(tensors, pred)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-d', '--data', type=str, help='data configuration file')
	parser.add_argument('-s', '--splits', type=str, default='0/10')
	parser.add_argument('-ax', '--n_axes', type=int, default=2)
	parser.add_argument('-dv', '--device', type=str, default='cpu')
	parser.add_argument('-vd', '--vocal_detail', action='store_true', help='vocal detail mode')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)

	if args.data:
		data_config = Configuration.createOrLoad(args.data, volatile=True)
		config['data'] = data_config['data']

	if args.splits is not None:
		config['data.splits'] = args.splits

	data, = loadDataset(config, data_dir=DATA_DIR, device=args.device)
	validator = Validator(config, args)

	validator.run(data)


if __name__ == '__main__':
	main()
