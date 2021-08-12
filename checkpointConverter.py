
import os
import argparse
import torch

from starry.utils.config import Configuration
from starry.utils import model_factory



def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='The source checkpoint file.')
	parser.add_argument('target', type=str, help='Target directory (with configuration).')

	args = parser.parse_args()

	config = Configuration.create(args.target) if args.target.endswith('.yaml') else Configuration(args.target)
	model = model_factory.loadModel(config['model'])

	checkpoint = torch.load(args.source, map_location='cpu')
	#print('checkpoint:', checkpoint['model_state_dict'].keys())

	# read data from checkpoint of old format
	epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['model_state_dict'])
	#print('model:', model)

	filename = os.path.basename(args.source)
	filename = os.path.splitext(filename)[0]

	output_path = config.localPath(f'{filename}.chkpt')

	torch.save({
		'epoch': epoch,
		'model': model.state_dict(),
	}, output_path)

	print('New checkpoint wrote:', output_path)


if __name__ == '__main__':
	main()
