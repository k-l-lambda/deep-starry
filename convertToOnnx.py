
import sys
import os
import logging
import torch
import argparse

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModel



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-s', '--shape', type=str, help='shape of input tensor, e.g. 1,3,256,256')
	parser.add_argument('-op', '--opset', type=int, default=11, help='ONNX opset version')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)
	model = loadModel(config['model'])

	name = 'untrained'
	if config['best']:
		name = os.path.splitext(config['best'])[0]

		checkpoint = torch.load(config.localPath(config['best']), map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		logging.info(f'checkpoint loaded: {config["best"]}')

	model.eval()
	model.no_overwrite = True

	outpath = config.localPath(f'{name}.onnx')
	dummy_input = torch.randn(*map(int, args.shape.split(',')))
	torch.onnx.export(model, dummy_input, outpath, verbose=True, input_names=['in'], output_names=['out'], opset_version=args.opset)

	logging.info(f'ONNX model saved to: {outpath}')


if __name__ == '__main__':
	main()
