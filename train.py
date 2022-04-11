
import os
import sys
import argparse
import logging

from starry.utils.config import Configuration
from starry.utils.trainer import Trainer
from starry.utils.dataset_factory import loadDataset



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	logging.basicConfig(format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%Y%m%d %H:%M:%S', level=logging.INFO,
		force=True, handlers=[
			logging.StreamHandler(sys.stdout),
			logging.FileHandler(config.localPath('trainer.log')),
		])

	logging.info('*	Loading data.')
	train, val = loadDataset(config, data_dir=DATA_DIR, device=config['trainer.device'])

	logging.info('*	Training.')
	trainer = Trainer(config)
	trainer.train(train, val)


if __name__ == '__main__':
	main()
