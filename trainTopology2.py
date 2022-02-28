
import os
import sys
import dill as pickle
import argparse
import logging

from starry.utils.config import Configuration
from starry.utils.trainer import Trainer
from starry.topology.data.semantics import DatasetScatter



DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-tr', '--truncate', type=int, default=None)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	logging.basicConfig(filename=config.localPath('trainer.log'),
		format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%Y%m%d %H:%M:%S', level=logging.INFO)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

	logging.info('Loading data.')
	data_path = os.path.join(DATA_DIR, config['data.file_name'])

	if data_path.endswith('.zip'):
		data_path = 'zip://' + data_path

	#meta = pickle.load(data_file)
	#config['model.args.d_model'] = meta['d_word']

	train, val = DatasetScatter.loadPackage(data_path, batch_size=config['data.batch_size'], splits=config['data.splits'], device=config['trainer.device'])

	if args.truncate is not None:
		train.entries = train.entries[:args.truncate]
		val.entries = val.entries[:args.truncate]

	logging.info('Training.')
	trainer = Trainer(config)
	trainer.train(train, val)


if __name__ == '__main__':
	main()
