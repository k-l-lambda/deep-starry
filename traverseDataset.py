
import sys
import os
import argparse
import logging
from tqdm import tqdm

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-s', '--splits', type=str, default='*0/1')
	parser.add_argument('-dv', '--device', type=str, default='cpu')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)
	if args.splits is not None:
		config['data.splits'] = args.splits

	data, = loadDataset(config, data_dir=DATA_DIR, device=args.device)

	for batch in tqdm(data, desc='Traversing dataset'):
		pass


if __name__ == '__main__':
	main()
