
import sys
import os
import logging
import argparse

from starry.utils.config import Configuration
from starry.stylegan.training_loop import training_loop



logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	logging.basicConfig(filename=config.localPath('log.txt'),
		format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%H:%M:%S', level=logging.INFO)

	training_loop(config)


if __name__ == '__main__':
	main()
