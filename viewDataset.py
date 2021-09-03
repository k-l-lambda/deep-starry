
import sys
import os
import argparse
import logging

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-s', '--splits', type=str, default='0/10')

	args = parser.parse_args()

	config = Configuration.create(args.config, volatile=True) if args.config.endswith('.yaml') else Configuration(args.config)
	if args.splits is not None:
		config['data.splits'] = args.splits

	data, = loadDataset(config, data_dir=VISION_DATA_DIR)
	print('data:', data)


if __name__ == '__main__':
	main()
