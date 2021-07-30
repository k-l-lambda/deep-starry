
import sys
import dill as pickle
import argparse
import logging

from starry.score_connection.data import Dataset
from starry.score_connection.trainer import Trainer



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data', type=str)
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

	datasets = Dataset.loadPackage(data, batch_size=args.batch_size, device=args.device, truncate=args.truncate)

	logging.info('Training.')
	trainer = Trainer(args)
	trainer.train(datasets['train'], datasets['val'])



if __name__ == '__main__':
	main()
