
import sys
import os
import logging
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import argparse

from starry.utils.config import Configuration
from starry.utils.model_factory import loadModel



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-l', '--lite', action='store_true', help='save model for lite interpreter')
	parser.add_argument('-m', '--mobile', action='store_true', help='optimize for mobile')
	parser.add_argument('-p', '--postfix', default='', help='postfix on model class name')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)
	model = loadModel(config['model'], postfix=args.postfix)

	name = 'untrained'
	if config['best']:
		if config['best'].endswith('.chkpt'):
			name = os.path.splitext(config['best'])[0]
		else:
			name = config['best']
			config['best'] += '.chkpt'

		checkpoint = torch.load(config.localPath(config['best']), map_location='cpu')
		if hasattr(model, 'deducer'):
			model.deducer.load_state_dict(checkpoint['model'], strict=False)
		else:
			model.load_state_dict(checkpoint['model'], strict=False)
		logging.info(f'checkpoint loaded: {config["best"]}')

	scriptedm = torch.jit.script(model)
	if args.mobile:
		scriptedm = optimize_for_mobile(scriptedm)
		name += '-mobile'

	outpath = config.localPath(f'{name}.ptl' if args.lite else f'{name}.pt')
	if args.lite:
		scriptedm._save_for_lite_interpreter(outpath)
	else:
		scriptedm.save(outpath)

	logging.info(f'Scripted model saved to: {outpath}')


if __name__ == '__main__':
	main()
