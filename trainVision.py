
import os
import sys
import dill as pickle
import argparse
import logging

from starry.utils.config import Configuration
from starry.utils.trainer import Trainer
from starry.utils.dataset_factory import loadDataset



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)

	args = parser.parse_args()

	config = Configuration.createdOrLoad(args.config)

	logging.info('*	Loading data.')
	train, val = loadDataset(config, data_dir=VISION_DATA_DIR, device=config['trainer.device'])

	logging.info('*	Training.')
	trainer = Trainer(config)
	trainer.train(train, val)


if __name__ == '__main__':
	main()
