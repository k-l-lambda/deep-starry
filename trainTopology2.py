
import os
import sys
import dill as pickle
import argparse
import logging

from starry.utils.config import Configuration
from starry.utils.trainer import Trainer
from starry.topology.data import Dataset



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-tr', '--truncate', type=int, default=None)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	logging.info('Loading data.')
	data_file = open(os.path.join(DATA_DIR, config['data.file_name']), 'rb')

	meta = pickle.load(data_file)
	config['model.args.d_model'] = meta['d_word']

	train, val = Dataset.loadPackage(data_file, batch_size=config['data.batch_size'], splits=config['data.splits'], device=config['trainer.device'])

	if args.truncate is not None:
		train.examples = train.examples[:args.truncate]
		val.examples = val.examples[:args.truncate]

	logging.info('Training.')
	trainer = Trainer(config)
	trainer.train(train, val)


if __name__ == '__main__':
	main()
