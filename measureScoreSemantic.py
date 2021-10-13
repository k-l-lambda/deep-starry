
import sys
import logging
import os
import yaml
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.utils.model_factory import loadModel



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


class MeasureDataset:
	def __init__(self, config, device='cuda'):
		super().__init__()

		self.model = loadModel(config['model'], postfix='Loss')

		checkpoint = torch.load(config.localPath(config['best']), map_location=device)
		self.model.deducer.load_state_dict(checkpoint['model'])
		logging.info(f'checkpoint loaded: {config["best"]}')

		self.model.to(device)
		self.model.eval()

		self.dataset, = loadDataset(config, data_dir=VISION_DATA_DIR, device=device)

	def run (self):
		with torch.no_grad():
			metric_data = {}

			for batch in tqdm(self.dataset, mininterval=2, desc='Measuring', leave=False):
				loss, metric = self.model(batch)

				for k, v in metric.items():
					metric_data[k] = metric_data[k] + v if k in metric_data else v

			self.stat = metric_data['semantic'].stat()
			#print('metric_data:', stat)

			return self.stat

	def saveConfidenceNormalization (self, filename):
		with open(filename, 'w') as file:
			table = list(map(lambda pair: {'semantic': pair[0], 'mean_confidence': pair[1]['confidence']}, self.stat['details'].items()))
			file.write(yaml.dump(table))


def main ():
	parser = ArgumentParser(description='Run the score semantic measuring')
	parser.add_argument('config', type=str)
	parser.add_argument('-d', '--data', type=str, help='data configuration file')
	parser.add_argument('-dv', '--device', type=str, default='cuda')
	parser.add_argument('-s', '--splits', type=str, default='0/10')
	parser.add_argument('-b', '--batch-size', type=int)
	parser.add_argument('-o', '--confidence-table', action='store_true', help='Output confidence normalization table')

	args = parser.parse_args()

	config = Configuration(args.config)

	if args.data:
		data_config = Configuration.createdOrLoad(args.data, volatile=True)
		config['data'] = data_config['data']

	config['data.splits'] = args.splits
	config['model.args.metric_quota'] = float('inf')

	if args.batch_size is not None:
		config['data.batch_size'] = args.batch_size

	def log (file, message):
		logging.info(message)
		file.write(message + '\n')

	measurer = MeasureDataset(config, device=args.device)

	with open(config.localPath(config['best'] + '.metric.log'), 'wt') as file:
		stat = measurer.run()

		details = list(stat['details'].items())
		log(file, f'{"LABEL":<24}\t{"ERRORS":<12}\tTRUE_COUNT\tACCURACY\tFEASIBILITY\tCONFIDENCE')
		lines = list(map(lambda item: f'{item[0]:<24}\t{item[1]["errors"]:<12}\t{item[1]["true_count"]}\t\t{item[1]["accuracy"]}\t\t{item[1]["feasibility"]:.3f}\t\t{item[1]["confidence"]:.4f}', details))
		log(file, '\n'.join(lines))

		log(file, f'accuracy: {stat["accuracy"]}')
		log(file, f'total_error_rate: {stat["total_error_rate"]}')
		log(file, f'total_true_count: {stat["total_true_count"]}')

	if args.confidence_table:
		confidence_path = config.localPath('confidence.yaml')
		measurer.saveConfidenceNormalization(confidence_path)

		logging.info('confidence table saved: %s', confidence_path)


if __name__ == '__main__':
	main()
