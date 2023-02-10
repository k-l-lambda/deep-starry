
import sys
import os
import argparse
import logging
import torch
from tqdm import tqdm

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.utils.predictor import Predictor
from starry.vision.stampViewer import StampViewer



# workaround cuda unavailable issue
torch.cuda.is_available()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('VISION_DATA_DIR') or os.environ.get('DATA_DIR')


class Validator (Predictor):
	def __init__ (self, config, args):
		super().__init__(device=args.device)

		self.viewer = StampViewer(config, cell_capacity=12)

		self.loadModel(config)


	def run (self, data):
		for batch, tensors in enumerate(tqdm(data)):
			#logging.info('batch: %d', batch)

			feature, label = tensors

			with torch.no_grad():
				pred = self.model(feature)
				#print('pred:', pred)

			for i, img in enumerate(feature):
				self.viewer.appendExample(img, label[i], pred[i])

		input("Press Enter to continue...")


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-d', '--data', type=str, help='data configuration file')
	parser.add_argument('-s', '--splits', type=str, default='0/10')
	parser.add_argument('-dv', '--device', type=str, default='cpu')

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
