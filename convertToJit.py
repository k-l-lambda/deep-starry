
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

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)
	model = loadModel(config['model'])

	name = 'untrained'
	if config['best']:
		name = os.path.splitext(config['best'])[0]

		checkpoint = torch.load(config.localPath(config['best']), map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		logging.info(f'checkpoint loaded: {config["best"]}')

	scriptedm = torch.jit.script(model)
	outpath = config.localPath(f'{name}.pt')
	torch.jit.save(scriptedm, outpath)

	logging.info(f'Scripted model saved to: {outpath}')


if __name__ == '__main__':
	main()
