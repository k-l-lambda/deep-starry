
import os
import sys
import argparse
import logging

import starry.utils.env
from starry.paraff.data.timewiseGraph import preprocessGraph



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('meta', type=str, help='paraff paragraph yaml file')
	parser.add_argument('source', type=str, help='semantic json directory path')

	args = parser.parse_args()

	logging.info('Building package from directory: %s', args.source)
	preprocessGraph(args.meta, args.source)

	logging.info('Done.')



if __name__ == '__main__':
	main()
