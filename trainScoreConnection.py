
import os
import sys
import dill as pickle
import argparse
import logging

from starry.utils.config import Configuration
from starry.score_connection.data import Dataset
from starry.score_connection.trainer import Trainer



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('--truncate', type=int, default=None)

	args = parser.parse_args()

	config = Configuration.create(args.config) if args.config.endswith('.yaml') else Configuration(args.config)

	logging.info('Loading data.')
	data = pickle.load(open(os.path.join(DATA_DIR, config['data.file_name']), 'rb'))
	config['model.args.d_model'] = data['d_word']

	train, val = Dataset.loadPackage(data, batch_size=config['data.batch_size'], splits=config['data.splits'], device=config['trainer.device'])

	if args.truncate > 0:
		train.examples = train.examples[:args.truncate]
		val.examples = val.examples[:args.truncate]

	logging.info('Training.')
	trainer = Trainer(config)
	trainer.train(train, val)


if __name__ == '__main__':
	main()