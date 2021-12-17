
import os
import sys
import argparse
import logging

from starry.utils.config import Configuration
from starry.utils.trainer import Trainer
from starry.utils.dataset_factory import loadDataset



VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	logging.basicConfig(filename=config.localPath('trainer.log'),
		format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%H:%M:%S', level=logging.INFO)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

	logging.info('*	Loading data.')
	train, val = loadDataset(config, data_dir=VISION_DATA_DIR, device=config['trainer.device'])

	logging.info('*	Training.')
	trainer = Trainer(config)
	trainer.train(train, val)


if __name__ == '__main__':
	main()
