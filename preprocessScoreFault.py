
import os
import sys
import argparse
import logging

from starry.utils.config import Configuration
from starry.vision.data.scoreFault import preprocessDataset



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='input data directory path')
	parser.add_argument('target', type=str, help='output path')
	parser.add_argument('-s', '--semantics', type=str, help='config file with semantic labels')

	args = parser.parse_args()

	semantic_config = Configuration.createOrLoad(args.semantics, volatile=True)
	semantics = semantic_config['data.args.labels']

	target = args.target
	if not os.path.isabs(target):
		target = os.path.join(DATA_DIR, target)

	logging.info('Building package from directory: %s', args.source)
	preprocessDataset(args.source, target, semantics=semantics)

	logging.info('Done.')



if __name__ == '__main__':
	main()
