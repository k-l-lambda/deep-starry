
import os
import sys
import dill as pickle
import argparse
import logging

from starry.score_connection.data import Dataset
from starry.score_connection.trainer import Trainer
from starry.utils.config import Configuration



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


DATA_DIR = os.environ.get('DATA_DIR')


'''def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data', type=str)
	parser.add_argument('-sp', '--splits', type=str, default='*1,2,3,4,5,6,7,8/10;9/10')
	parser.add_argument('-b', '--batch_size', type=int, default=16)
	parser.add_argument('-dv', '--device', type=str, default='cuda')
	parser.add_argument('-sv', '--save_mode', type=str, default='best')
	parser.add_argument('-e', '--epoch', type=int, default=400)
	parser.add_argument('-lr', '--lr_mul', type=float, default=2.0)
	parser.add_argument('-warm', '--n_warmup_steps', type=int, default=4000)
	parser.add_argument('--truncate', type=int, default=None)

	args = parser.parse_args()
	args.output_dir = './output'

	logging.info('Loading data.')
	data = pickle.load(open(args.data, 'rb'))
	args.d_model = data['d_word']

	train, val = Dataset.loadPackage(data, batch_size=args.batch_size, splits=args.splits, device=args.device)

	if args.truncate > 0:
		train.examples = train.examples[:args.truncate]
		val.examples = val.examples[:args.truncate]

	logging.info('Training.')
	trainer = Trainer(args)
	trainer.train(train, val)'''
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
