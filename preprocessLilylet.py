import argparse
import logging
import os
import sys

from starry.lilylet.data.patchifier import pack_lilylet_notagen
from starry.utils.config import Configuration


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

DATA_DIR = os.environ.get('DATA_DIR', '.')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str, help='config of Lilylet data to preprocess')
	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)
	preprocess_args = config['data.args'].get('preprocess', {})
	source_dir = preprocess_args.get('source_dir')
	if source_dir is None:
		source_dir = os.path.join(DATA_DIR, preprocess_args.get('source', config['data.root']))
	elif not os.path.isabs(source_dir):
		source_dir = os.path.join(DATA_DIR, source_dir)

	output = preprocess_args.get('output', config['data.root'])
	output_path = output if os.path.isabs(output) else os.path.join(DATA_DIR, output)

	tokenizer_path = config['data.args']['tokenizer_path']
	patch_size = config['data.args'].get('patch_size', 16)
	patch_length = config['data.args'].get('patch_length', 2048)
	patch_stream = config['data.args'].get('patch_stream', True)

	logging.info('Preprocessing Lilylet data: %s', source_dir)
	artifact = pack_lilylet_notagen(
		source_dir=source_dir,
		output_path=output_path,
		tokenizer_path=tokenizer_path,
		patch_size=patch_size,
		patch_length=patch_length,
		patch_stream=patch_stream,
	)
	logging.info('Wrote %s', output_path)
	logging.info('Files: %s', artifact['stats']['files'])
	logging.info('Unknown total: %s', artifact['stats']['unknown_total'])


if __name__ == '__main__':
	main()
