
import os
import sys
import argparse
import logging

import starry.utils.env
from starry.paraff.midiseq import packMidiseqYaml



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='input yaml file path')

	args = parser.parse_args()

	packMidiseqYaml(args.source)

	logging.info('Done.')



if __name__ == '__main__':
	main()
