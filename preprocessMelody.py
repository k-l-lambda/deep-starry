
import os
import sys
import argparse
import logging

import starry.utils.env
from starry.melody.data.preprocess import preprocessDataset, preprocessDatasetFrames



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='input data directory path')
	parser.add_argument('target', type=str, help='output path')
	parser.add_argument('-f', '--frames', action='store_true', help='frames mode')

	args = parser.parse_args()

	target = args.target
	if not os.path.isabs(target):
		target = os.path.join(DATA_DIR, target)

	logging.info('Building package from directory: %s', args.source)
	if args.frames:
		preprocessDatasetFrames(args.source, target)
	else:
		preprocessDataset(args.source, target)

	logging.info('Done.')



if __name__ == '__main__':
	main()
