
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
	parser.add_argument('-s', '--shapes', type=str, help='shapes of input tensors, e.g. 1,3,256,256;1,16')
	parser.add_argument('-op', '--opset', type=int, default=11, help='ONNX opset version')
	parser.add_argument('-in', '--input_names', type=str, default='in', help='e.g. in1;in2')
	parser.add_argument('-out', '--output_names', type=str, default='out', help='e.g. out1;out2')

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

	shapes = args.shapes.split(';')
	dummy_inputs = tuple(torch.randn(*map(int, shape.split(','))) for shape in shapes)
	torch.onnx.export(model, dummy_inputs, outpath,
		verbose=True,
		input_names=args.input_names.split(';'),
		output_names=args.output_names.split(';'),
		opset_version=args.opset)

	logging.info(f'ONNX model saved to: {outpath}')


if __name__ == '__main__':
	main()
