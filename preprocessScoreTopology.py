
import os
import sys
import argparse
import logging
import re

import starry.utils.env
from starry.topology.data import preprocessDataset, preprocessDatasetScatter



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='input data directory path')
	parser.add_argument('target', type=str, help='output path')
	parser.add_argument('-seq', '--n_seq_max', type=int, default=0x100)
	parser.add_argument('-d', '--d_word', type=int, default=0x200)
	parser.add_argument('-n', '--name_id', type=str, default=r'^(.+)\.\d+\.[\w.]+$')
	parser.add_argument('-v1', action='store_true')
	parser.add_argument('-a', '--n_augment', type=int, default=64)

	args = parser.parse_args()

	target = args.target
	if not os.path.isabs(target):
		target = os.path.join(DATA_DIR, target)

	logging.info('Building package from directory: %s', args.source)
	if args.v1:
		with open(target, 'wb') as file:
			preprocessDataset(args.source, file, name_id=re.compile(args.name_id), n_seq_max=args.n_seq_max, d_word=args.d_word)
	else:
		preprocessDatasetScatter(args.source, target, name_id=re.compile(args.name_id), n_seq_max=args.n_seq_max, d_word=args.d_word, n_augment=args.n_augment)

	logging.info('Done.')



if __name__ == '__main__':
	main()
