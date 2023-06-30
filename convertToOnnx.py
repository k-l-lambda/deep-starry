
import sys
import os
import logging
import torch
import argparse

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModel, registerModels
from onnxTypecast import convert_model_to_int32



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def runConfig (onnx_config, model_loader, outpath):
	input_names = [input['name'] for input in onnx_config['inputs']]
	output_names = onnx_config['outputs']

	#shapes = [tuple(onnx_config['inputs'][name]) for name in input_names]
	dummy_inputs = tuple(torch.zeros(*input['shape'], dtype=getattr(torch, input.get('dtype', 'float32'))) for input in onnx_config['inputs'])
	opset = onnx_config['opset']

	truncate_long = onnx_config.get('truncate_long')
	temp_path = outpath.replace('.onnx', '.temp.onnx')

	model_postfix = onnx_config.get('model_postfix', '')
	model = model_loader(model_postfix)

	torch.onnx.export(model, dummy_inputs, temp_path if truncate_long else outpath,
		verbose=True,
		input_names=input_names,
		output_names=output_names,
		opset_version=opset)

	if truncate_long:
		convert_model_to_int32(temp_path, outpath)

	logging.info(f'ONNX model saved to: {outpath}')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-s', '--shapes', type=str, help='shapes of input tensors, e.g. 1,3,256,256;1,16')
	parser.add_argument('-op', '--opset', type=int, default=11, help='ONNX opset version')
	parser.add_argument('-in', '--input_names', type=str, default='in', help='e.g. in1;in2')
	parser.add_argument('-out', '--output_names', type=str, default='out', help='e.g. out1;out2')

	args = parser.parse_args()

	registerModels()
	from starry.utils.model_factory import model_dict

	config = Configuration.createOrLoad(args.config)

	name = 'untrained'
	if config['best']:
		name = os.path.splitext(config['best'])[0]

	def loadModel_ (postfix):
		model = loadModel(config['model'], postfix=postfix)

		if config['best']:
			checkpoint = torch.load(config.localPath(config['best']), map_location='cpu')
			if hasattr(model, 'deducer'):
				model.deducer.load_state_dict(checkpoint['model'], strict=False)
			else:
				model.load_state_dict(checkpoint['model'])
			logging.info(f'checkpoint loaded: {config["best"]}')

		model.eval()
		model.no_overwrite = True

		return model

	if args.shapes is not None:
		model_postfix = config['onnx.postfix'] or ('Onnx' if (config['model.type'] + 'Onnx' in model_dict) else '')
		model = loadModel_(model_postfix)

		truncate_long = config['onnx.truncate_long_tensor']
		out_name = f'{name}.temp.onnx' if truncate_long else f'{name}.onnx'
		outpath = config.localPath(out_name)

		opset = args.opset

		shapes = args.shapes.split(';')
		shapes = [tuple(map(int, shape.split(','))) for shape in shapes]
		input_names = args.input_names.split(';')
		output_names = args.output_names.split(';')
		dummy_inputs = tuple(torch.randn(*shape) for shape in shapes)

		torch.onnx.export(model, dummy_inputs, outpath,
			verbose=True,
			input_names=input_names,
			output_names=output_names,
			opset_version=opset)

		if truncate_long:
			temp_path = outpath
			outpath = config.localPath(f'{name}.onnx')
			convert_model_to_int32(temp_path, outpath)

		logging.info(f'ONNX model saved to: {outpath}')
	elif config['onnx']:
		if 'multiple' in config['onnx']:
			for key, onnx_config in config['onnx.multiple'].items():
				runConfig(onnx_config, loadModel_, outpath=config.localPath(f'{name}-{key}.onnx'))
		else:
			runConfig(config['onnx'], loadModel_, outpath=config.localPath(f'{name}.onnx'))


if __name__ == '__main__':
	main()
