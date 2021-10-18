
import sys
import os
import argparse
import logging

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.vision.validator import Validator



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-d', '--data', type=str, help='data configuration file')
	parser.add_argument('-s', '--splits', type=str, default='0/10')
	parser.add_argument('-k', '--skip', action='store_true', help='skip perfect samples')
	parser.add_argument('-g', '--gauge', action='store_true', help='gauge mode')
	parser.add_argument('-ns', '--no-splice', action='store_true', help='avoid calling splice_pieces')

	args = parser.parse_args()

	config = Configuration(args.config)

	if args.data:
		data_config = Configuration.createdOrLoad(args.data, volatile=True)
		config['data'] = data_config['data']

	if args.splits is not None:
		config['data.splits'] = args.splits

	data, = loadDataset(config, data_dir=VISION_DATA_DIR)
	validator = Validator(config, skip_perfect=args.skip, gauge_mode=args.gauge, splice=not args.no_splice)

	validator.run(data)


if __name__ == '__main__':
	main()
