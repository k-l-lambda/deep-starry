
import sys
import os
import argparse
import torch
import logging
from tqdm import tqdm

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.utils.model_factory import loadModel
from starry.utils.trainer import print_metric, stat_average



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('VISION_DATA_DIR') or os.environ.get('DATA_DIR')


class Validator:
	def __init__(self, config, device):
		super().__init__()

		#self.config = config
		self.model = loadModel(config['model'], postfix='Loss')
		self.model.to(device)

		if config['best']:
			weights_path = config.localPath(config['best'])
			checkpoint = torch.load(weights_path, map_location=device)
			self.model.deducer.load_state_dict(checkpoint['model'])

			logging.info('checkpoint loaded: %s', weights_path)

		self.model.eval().requires_grad_(False)

	def run(self, dataset):
		total_loss, n_batch = 0, 0
		metric_data = {}

		with torch.no_grad():
			for batch in tqdm(dataset, desc='Measuring'):
				# forward
				loss, metric = self.model(batch)

				n_batch += 1
				total_loss += loss.item()

				metric = metric if type(metric) == dict else {'acc': metric}
				for k, v in metric.items():
					metric_data[k] = (metric_data[k] + v) if k in metric_data else v

		stat = self.model.stat if hasattr(self.model, 'stat') else stat_average
		metrics = stat(metric_data, n_batch)

		loss = total_loss / n_batch

		logging.info('loss: %f', loss)
		logging.info('metric: %s', print_metric(metrics))


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-d', '--data', type=str, required=True, help='data configuration file')
	parser.add_argument('-s', '--splits', type=str, default='0/10')
	parser.add_argument('-dv', '--device', type=str, default='cpu')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)

	if args.data:
		data_config = Configuration.createOrLoad(args.data, volatile=True)
		config['data'] = data_config['data']
		if config['data.args_variant.1']:
			config['data.args_variant.0'] = config['data.args_variant.1']

	if args.splits is not None:
		config['data.splits'] = args.splits

	data, = loadDataset(config, data_dir=DATA_DIR, device=args.device)
	validator = Validator(config, device=args.device)

	validator.run(data)


if __name__ == '__main__':
	main()
