
import sys
import os
import argparse
import logging
import torch
from tqdm import tqdm

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.utils.predictor import Predictor



# workaround cuda unavailable issue
torch.cuda.is_available()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


def mse (pred, target):
	return torch.mean((pred - target) ** 2)


def mseGreen (pred, target):
	return torch.mean((pred[:, 1] - target[:, 1]) ** 2)


class Checker (Predictor):
	def __init__(self, config, device='cuda'):
		super().__init__(device=device)

		self.config = config
		self.loadModel(config)


	def run(self, dataset):
		results = []

		with torch.no_grad():
			for name, feature, target in tqdm(dataset, desc='Checking'):
				pred = self.model(feature)
				differ = mseGreen(pred, target).item()
				#logging.info('differ: %s, %.4f', name, differ.item())

				if differ > 0.01:
					results.append((name, differ))

				if differ > 0.1:
					logging.info('differ: %s, %.4f', name, differ)

		results.sort(key=lambda x: -x[1])
		text = '\n'.join([f'{x[0]}:\t{x[1]}' for x in results])
		logging.info('results: %s', text)

		with open(self.config.localPath('checker.txt'), 'w') as file:
			file.write(text)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-d', '--data', type=str, help='data configuration file')
	parser.add_argument('-s', '--splits', type=str, default='0/1')
	parser.add_argument('-dv', '--device', type=str, default='cuda')

	args = parser.parse_args()

	config = Configuration(args.config)

	if args.data:
		data_config = Configuration.createOrLoad(args.data, volatile=True)
		config['data'] = data_config['data']

	data, = loadDataset(config, data_dir=VISION_DATA_DIR, device=args.device, splits=args.splits)
	checker = Checker(config, device=args.device)
	checker.run(data)


if __name__ == '__main__':
	main()
