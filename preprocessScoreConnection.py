
import sys
import argparse
import dill as pickle
import logging

from starry.score_connection.data import preprocessDataset



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='input data directory path')
	parser.add_argument('target', type=str, help='output path')
	parser.add_argument('-seq', '--n_seq_max', type=int, default=0x100)
	parser.add_argument('-d', '--d_word', type=int, default=0x200)

	args = parser.parse_args()

	logging.info('Building package from directory: %s', args.source)
	data = preprocessDataset(args.source, n_seq_max=args.n_seq_max, d_word=args.d_word)

	logging.info('Writing package: %s', args.target)
	with open(args.target, 'wb') as file:
		pickle.dump(data, file)

	logging.info('Done.')



if __name__ == '__main__':
	main()
