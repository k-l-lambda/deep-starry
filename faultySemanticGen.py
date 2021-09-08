
import sys
import os
import argparse
import logging
import torch
import numpy as np
import yaml

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.utils.predictor import Predictor
from starry.vision import contours
from starry.vision.images import MARGIN_DIVIDER, splicePieces
from starry.vision.data import GraphScore
from starry.vision.score_semantic import ScoreSemantic



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


class FaultyGenerator (Predictor):
	def __init__(self, config, device):
		super().__init__()

		self.config = config
		self.loadModel(config)

		self.compounder = contours.Compounder(config)

	def run(self, dataset):
		labels = self.config['data.args.labels']
		unit_size = self.config['data.args.unit_size']

		with torch.no_grad():
			for name, source, graph in dataset:
				print('name:', name)
				#print('source:', source.shape)
				#print('graph:', graph)
				#print('graph:', len(graph['points']))

				pred = self.model(source)
				#print('pred:', pred.shape)
				heatmap = splicePieces(pred.cpu().numpy(), MARGIN_DIVIDER, keep_margin=True)
				heatmap = np.uint8(heatmap * 255)
				#print('heatmap:', heatmap.shape)

				semantics = ScoreSemantic(heatmap, labels)
				semantics.discern(graph)
				#print('semantics:', len(semantics.json()['points']))

				fake_positive = len([p for p in semantics.data['points'] if p['value'] == 0 and p['confidence'] >= 1])
				fake_negative = len([p for p in semantics.data['points'] if p['value'] > 0 and p['confidence'] < 1])

				error_rate = (fake_positive + fake_negative) / len(graph['points'])
				print('error_rate:', error_rate, fake_positive, fake_negative)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('data', type=str)
	parser.add_argument('-m', '--multiple', type=int, default=1, help='how many samples for one staff')
	parser.add_argument('-d', '--device', type=str, default='cuda')
	#parser.add_argument('-b', '--batch-size', type=int, default=1)

	args = parser.parse_args()

	data_config = Configuration.create(args.data, volatile=True)

	config = Configuration(args.config)
	config['data'] = data_config['data']
	config['data.splits'] = '0/1'
	#config['data.batch_size'] = args.batch_size

	#data, = loadDataset(config, data_dir=VISION_DATA_DIR, device=args.device)
	root = os.path.join(VISION_DATA_DIR, config['data.root'])
	dataset = GraphScore(root, device=args.device, multiple=args.multiple, **config['data.args'])
	generator = FaultyGenerator(config, device=args.device)
	generator.run(iter(dataset))


if __name__ == '__main__':
	main()
