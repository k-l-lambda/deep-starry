
import sys
import os
import argparse
import logging
import torch
import tempfile

from starry.utils.config import Configuration
from starry.utils.trainerQuantitative import Trainer



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-b', '--backend', type=str, default='gloo')
	parser.add_argument('-c', '--config_only', action='store_true', help='create config, no training')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	if args.config_only:
		print('Config created:', config.dir)
		return

	with tempfile.TemporaryDirectory(dir=config.dir) as temp_dir:
		init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))

		torch.multiprocessing.set_start_method('spawn')
		torch.multiprocessing.spawn(fn=Trainer.run, args=(config, VISION_DATA_DIR, init_file, args.backend), nprocs=Trainer.PROC_COUNT)


if __name__ == '__main__':
	main()
