
import os
import argparse
#import logging
import torch

from starry.utils.config import Configuration
from starry.utils.trainerQuantitative import Trainer
#from starry.utils.dataset_factory import loadDataset



VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-b', '--backend', type=str, default='nccl')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	'''logging.info('*	Loading data.')
	train, val = loadDataset(config, data_dir=VISION_DATA_DIR, device=config['trainer.device'])

	logging.info('*	Training.')
	trainer = Trainer(config)
	#trainer.train(train)
	trainer.validate(val)'''
	torch.multiprocessing.set_start_method('spawn')
	torch.multiprocessing.spawn(fn=Trainer.run, args=(config, VISION_DATA_DIR, args.backend), nprocs=Trainer.PROC_COUNT)


if __name__ == '__main__':
	main()
