
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
from starry.vision.score_semantic import ScoreSemanticDual



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


class SubPredictor:
	def __init__ (self, config, device='cpu', labels=None):
		self.model = loadModel(config['model'], postfix='Loss')

		checkpoint_path = config['best'] if config['best'].endswith('.chkpt') else config['best'] + '.chkpt'
		checkpoint = torch.load(config.localPath(checkpoint_path), map_location=device)
		self.model.deducer.load_state_dict(checkpoint['model'])
		logging.info(f'checkpoint loaded: {checkpoint_path}')

		self.model.to(device)
		self.model.eval()

		self.channels = torch.tensor([labels.index(name) for name in config['data.args.labels']]).long().to(device)


	def __call__ (self, batch):
		feature, target = batch
		target = target.index_select(1, self.channels)
		return self.model((feature, target))


class MeasureDatasetCluster:
	def __init__(self, config, device='cuda'):
		super().__init__()

		labels = config['data.args.labels']

		sub_configs = [Configuration(config.localPath(dirname)) for dirname in config['subs']]
		self.predictors = [SubPredictor(cfg, device=device, labels=labels) for cfg in sub_configs]

		self.dataset, = loadDataset(config, data_dir=VISION_DATA_DIR, device=device)

	def run (self):
		with torch.no_grad():
			semantics = [None for _ in self.predictors]

			for batch in tqdm(self.dataset, mininterval=2, desc='Measuring', leave=False):
				for i, predictor in enumerate(self.predictors):
					loss, metric = predictor(batch)

					#for k, v in metric.items():
					#	metric_data[k] = metric_data[k] + v if k in metric_data else v
					semantic = metric['semantic']
					semantics[i] = semantic if semantics[i] is None else semantics[i] + semantic

			ss = ScoreSemanticDual.merge_layers(semantics)
			self.stat = ss.stat()

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

	logging.basicConfig(filename=config.localPath(f'measureScoreSemantic-{config["best"] or ""}.log'),
		format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%Y%m%d %H:%M:%S', level=logging.INFO)
	logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

	if args.data:
		logging.info('Data: %s', args.data)
		data_config = Configuration.createOrLoad(args.data, volatile=True)
		config['data'] = data_config['data']

	config['data.splits'] = args.splits
	config['model.args.metric_quota'] = float('inf')

	if args.batch_size is not None:
		config['data.batch_size'] = args.batch_size

	MeasurerClass = MeasureDatasetCluster if config['subs'] else MeasureDataset
	measurer = MeasurerClass(config, device=args.device)

	stat = measurer.run()

	details = list(stat['details'].items())
	logging.info(f'\n{"LABEL":<24}\t{"ERRORS":<12}\tTRUE_COUNT\tACCURACY\tFEASIBILITY\tCONFIDENCE')
	lines = list(map(lambda item: f'{item[0]:<24}\t{item[1]["errors"]:<12}\t{item[1]["true_count"]}\t\t{item[1]["accuracy"]}\t\t{item[1]["feasibility"]:.3f}\t\t{item[1]["confidence"]:.4f}', details))
	logging.info('\n' + '\n'.join(lines))

	logging.info(f'accuracy: {stat["accuracy"]}')
	logging.info(f'total_error_rate: {stat["total_error_rate"]}')
	logging.info(f'total_true_count: {stat["total_true_count"]}')

	if args.confidence_table:
		confidence_path = config.localPath('confidence.yaml')
		measurer.saveConfidenceNormalization(confidence_path)

		logging.info('confidence table saved: %s', confidence_path)


if __name__ == '__main__':
	main()
