
import sys
import os
import argparse
import logging

from starry.utils.config import Configuration
from starry.vision.scoreSemanticCropper import ScoreSemanticCropper



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('VISION_DATA_DIR')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-o', '--output', type=str, help='output data dir')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)
	cropper = ScoreSemanticCropper(config, DATA_DIR)

	cropper.run(self.output)


if __name__ == '__main__':
	main()
