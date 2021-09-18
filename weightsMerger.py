
import sys
import os
from argparse import ArgumentParser
import torch
import logging

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModelAndWeights



logging.basicConfig(stream=sys.stdout, level=logging.INFO)



if __name__ == "__main__":
	parser = ArgumentParser(description = 'merge widgets model weights')
	parser.add_argument('main', type=str, help='main model config file')
	parser.add_argument('increment', type=str, help='appended weights config folder')
	parser.add_argument('-o', '--output', type=str, default='pretrained', help='output pkl file name')

	args = parser.parse_args()

	main = Configuration.createdOrLoad(args.main)
	increment = Configuration(args.increment)

	main_model, checkpoint = loadModelAndWeights(main, main['best'])
	increment_model, _ = loadModelAndWeights(increment, increment['best'])

	main_model.load_state_dict(increment_model.state_dict())
	checkpoint['model'] = main_model.state_dict()
	checkpoint['epoch'] = checkpoint.get('epoch', -1)

	output_filename = f'{args.output}.chkpt'
	output_path = main.localPath(output_filename)
	torch.save(checkpoint, output_path)

	main['trainer.pretrained_weights'] = output_filename
	main.save()

	logging.info('Weights saveing done: %s', output_path)
