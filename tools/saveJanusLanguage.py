
import os
import sys
import argparse
import logging
import torch

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModelAndWeights


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	model, cp = loadModelAndWeights(config, config['best'])
	model.to(torch.bfloat16)

	output_path = config.localPath(config['best'][:-6])
	model.save_pretrained(output_path)

	logging.info('Checkpoint saved to %s', output_path)


if __name__ == '__main__':
	main()
