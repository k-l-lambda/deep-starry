
import os
import argparse
#import logging
import torch

from starry.utils.config import Configuration
from starry.utils.trainerQuantitative import Trainer



VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-b', '--backend', type=str, default='nccl')
	parser.add_argument('-c', '--config_only', action='store_true', help='create config, no training')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	if args.config_only:
		print('Config created:', config.dir)
		return

	torch.multiprocessing.set_start_method('spawn')
	torch.multiprocessing.spawn(fn=Trainer.run, args=(config, VISION_DATA_DIR, args.backend), nprocs=Trainer.PROC_COUNT)


if __name__ == '__main__':
	main()
