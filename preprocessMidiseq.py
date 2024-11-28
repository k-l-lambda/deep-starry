
import os
import sys
import argparse
import logging

from starry.utils.config import Configuration
from starry.paraff.midiseq import packMidiseqYaml, summaryMeasures



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str, help='config of data to preprocess')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	source_base = os.path.join(DATA_DIR, config['data.root'])
	packMidiseqYaml(source_base + '.midiseq.yaml')

	encoder_config = config['data.args.paraff_encoder']
	n_seq = config['data.args.n_seq_paraff']
	summaryMeasures(source_base + '.midiseq.paraff', n_seq, encoder_config)


	logging.info('Done.')



if __name__ == '__main__':
	main()
