
import sys
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
	parser.add_argument('-p', '--postfix', type=str, default='', help='main model class postfix')
	parser.add_argument('-m', '--load_method', type=str, help='state load method, default is "load_state_dict"')

	args = parser.parse_args()

	main = Configuration.createOrLoad(args.main)
	increment = Configuration(args.increment)

	logging.basicConfig(format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%Y%m%d %H:%M:%S', level=logging.INFO,
		force=True, handlers=[
			logging.StreamHandler(sys.stdout),
			logging.FileHandler(main.localPath('trainer.log')),
		])

	main_model, checkpoint = loadModelAndWeights(main, main['best'], postfix=args.postfix)
	increment_model, _ = loadModelAndWeights(increment, increment['best'])

	if args.load_method:
		load = getattr(main_model, args.load_method)
		load(increment_model)
	else:
		main_model.load_state_dict(increment_model.state_dict())
	checkpoint['model'] = (main_model.deducer if args.postfix == 'Loss' else main_model).state_dict()
	checkpoint['epoch'] = checkpoint.get('epoch', -1)

	output_filename = f'{args.output}.chkpt'
	output_path = main.localPath(output_filename)
	torch.save(checkpoint, output_path)

	main['trainer.pretrained_weights'] = output_filename
	main.save()

	logging.info('Weights saving done: %s', output_path)
