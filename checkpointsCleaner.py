
import sys
import os
import logging
import re

from starry.utils.config import Configuration



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main (root_path):
	for root, dirs, files in os.walk(root_path):
		#if len(files):
		#	print('dirs:', root, files)
		if '.state.yaml' in files:
			try:
				config = Configuration(root)
				if config['best'] and (config['best'] in files):
					#print('config:', root, config['best'])
					files_to_delete = [name for name in files if name != config['best'] and re.match(r'^model.*\.chkpt$', name)]
					#print('files_to_delete:', root, config['best'], files_to_delete)
					for name in files_to_delete:
						os.remove(os.path.join(root, name))
					logging.info('%s: %d files deleted.', root, len(files_to_delete))
			except:
				logging.warning('error in configuration loading: %s', sys.exc_info()[1])


if __name__ == '__main__':
	main(sys.argv[1])
