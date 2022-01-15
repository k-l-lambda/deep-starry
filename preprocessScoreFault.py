
import os
import sys
import argparse
import logging

import starry.utils.env
from starry.vision.data.scoreFault import preprocessDataset



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='input data directory path')
	parser.add_argument('target', type=str, help='output path')
	parser.add_argument('-seq', '--n_seq_max', type=int, default=0x100)
	parser.add_argument('-d', '--d_word', type=int, default=0x200)

	args = parser.parse_args()

	target = args.target
	if not os.path.isabs(target):
		target = os.path.join(DATA_DIR, target)

	logging.info('Building package from directory: %s', args.source)
	preprocessDataset(args.source, target, n_seq_max=args.n_seq_max, d_word=args.d_word)

	logging.info('Done.')



if __name__ == '__main__':
	main()
