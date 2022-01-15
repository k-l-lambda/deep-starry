
import sys
import os
import argparse
import logging
import torch
import numpy as np
import yaml
import json
import zlib
from tqdm import tqdm

from starry.utils.config import Configuration
from starry.utils.predictor import Predictor
from starry.vision.images import MARGIN_DIVIDER, splicePieces
from starry.vision.data import GraphScore
from starry.vision.score_semantic import ScoreSemantic
from starry.vision.data.score import FAULT, FAULT_TARGET
from starry.vision.data.renderScore import renderTargetFromGraph



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


VISION_DATA_DIR = os.environ.get('VISION_DATA_DIR')


class FaultyGenerator (Predictor):
	def __init__ (self, config, root, load_confidence_path=False, by_render=False, device='cpu'):
		super().__init__(device=device)

		root = os.path.splitext(root)[0] if root.endswith('.zip') else root
		self.root = os.path.join(VISION_DATA_DIR, root)
		self.config = config

		if not by_render:
			self.loadModel(config)

		self.by_render = by_render
		self.output_dir = FAULT_TARGET if by_render else FAULT

		os.makedirs(os.path.join(self.root, self.output_dir), exist_ok=True)

		self.confidence_table = None
		if load_confidence_path:
			confidence_path = config.localPath('confidence.yaml')
			if os.path.exists(confidence_path):
				with open(confidence_path, 'r') as file:
					self.confidence_table = yaml.safe_load(file)
					logging.info('confidence_table loaded: %s', confidence_path)

	def run (self, dataset):
		labels = self.config['data.args.labels']
		unit_size = self.config['data.args.unit_size']

		total_true_positive, total_true_negative, total_fake_positive, total_fake_negative = 0, 0, 0, 0

		iterator = iter(dataset)

		with torch.no_grad():
			with tqdm(total=len(dataset)) as progress_bar:
				for name, source, graph in iterator:
					#print('name:', name)
					#print('source:', source.shape)
					#print('graph:', graph)
					#print('graph:', len(graph['points']))

					heatmap = None
					if self.by_render:
						heatmap = renderTargetFromGraph(graph, labels, (source.shape[2], source.shape[0] * source.shape[3]), unit_size=unit_size, name=name)
						heatmap = np.moveaxis(heatmap, -1, 0)
					else:
						pred = self.model(source)
						#print('pred:', pred.shape)
						heatmap = splicePieces(pred.cpu().numpy(), MARGIN_DIVIDER, pad_margin=True)
						heatmap = np.uint8(heatmap * 255)
					#print('heatmap:', heatmap.shape, source.shape)

					semantics = ScoreSemantic(heatmap, labels, confidence_table=self.confidence_table)
					semantics.discern(graph)
					#print('semantics:', len(semantics.json()['points']))

					fake_positive = len([p for p in semantics.data['points'] if p['value'] == 0 and p['confidence'] >= 1])
					fake_negative = len([p for p in semantics.data['points'] if p['value'] > 0 and p['confidence'] < 1])
					true_positive = len([p for p in semantics.data['points'] if p['value'] > 0 and p['confidence'] >= 1])
					true_negative = len([p for p in semantics.data['points'] if p['value'] == 0 and p['confidence'] < 1])

					total_true_positive += true_positive
					total_true_negative += true_negative
					total_fake_positive += fake_positive
					total_fake_negative += fake_negative

					error_rate = (fake_positive + fake_negative) / max(1, true_positive + fake_negative)
					noise_rate = true_negative / max(1, true_positive + fake_negative)
					#logging.info('error rate: %.4f, %.4f', error_rate, noise_rate)
					progress_bar.set_description(f'{name:<20}, err: {error_rate:.4f}, noise: {noise_rate:.4f}')

					if noise_rate < 1 and error_rate > 0:
						semantics.data['points'].sort(key=lambda p: p['x'])

						self.saveFaultGraph(name, semantics.json())

					progress_bar.update()

		error_rate = (total_fake_positive + total_fake_negative) / max(1, total_true_positive + total_fake_negative)

		with open(os.path.join(self.root, self.output_dir, '.stat.yaml'), 'w') as file:
			yaml.dump({
				'total_true_positive': total_true_positive,
				'total_true_negative': total_true_negative,
				'total_fake_positive': total_fake_positive,
				'total_fake_negative': total_fake_negative,
				'error_rate': error_rate,
			}, file)

	def saveFaultGraph (self, name, graph):
		content = json.dumps(graph)
		hash = zlib.crc32(content.encode('utf8'))
		hash = '{0:0{1}x}'.format(hash, 8)
		#print('fault:', self.root, name, hash)

		path = os.path.join(self.root, self.output_dir, f'{name}.{hash}.json')
		with open(path, 'w') as file:
			file.write(content)

		#logging.info('fault saved: %s', path)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('data', type=str)
	parser.add_argument('-m', '--multiple', type=int, default=1, help='how many samples for one staff')
	parser.add_argument('-dv', '--device', type=str, default='cuda')
	parser.add_argument('-s', '--split', type=str, default='0/1')
	parser.add_argument('-r', '--render', action='store_true', help='by rendering target, rather than model prediciton')

	args = parser.parse_args()

	data_config = Configuration.create(args.data, volatile=True)

	config = Configuration(args.config)
	config['data'] = data_config['data']
	#config['data.splits'] = args.splits

	root = os.path.join(VISION_DATA_DIR, config['data.root'])
	dataset = GraphScore(root, shuffle=False, device=args.device, split=args.split, multiple=args.multiple, **config['data.args'])
	generator = FaultyGenerator(config, root=config['data.root'], by_render=args.render, device=args.device)
	generator.run(dataset)


if __name__ == '__main__':
	main()
