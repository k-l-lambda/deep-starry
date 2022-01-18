
import sys
import os
import numpy as np
import torch
import yaml
import logging
import PIL.Image
import traceback

from ..utils.config import Configuration
from ..utils.predictor import Predictor
from .images import sliceFeature, spliceOutputTensor, MARGIN_DIVIDER
from . import transform
from .score_semantic import ScoreSemantic, ScoreSystem



BATCH_SIZE = int(os.environ.get('SCORE_SEMANTIC_PROCESSOR_BATCH_SIZE', '1'))

PATH_BATCH_SIZE = BATCH_SIZE * 4


def loadImage (path):
	image = PIL.Image.open(open(path, 'rb')).convert('L')
	arr = np.array(image)
	if len(arr.shape) == 2:
		arr = np.expand_dims(arr, axis=2)

	return arr


class ScoreSemanticProcessor (Predictor):
	def __init__(self, config, device='cpu', inspect=False):
		super().__init__(device=device)

		self.inspect = inspect
		if inspect:
			config['model.type'] = config['model.type'] + 'Inspection'

		self.loadModel(config)

		data_args = config['data.args'] or config['data']

		self.slicing_width = data_args['slicing_width']
		self.labels = data_args['labels']

		if config['rectifier']:
			self.semantics = ['_PAD', *self.labels]

			rectifier_config = Configuration(os.path.join(config.dir, '..', config['rectifier']))
			self.n_seq_max = rectifier_config['data.args.n_seq_max']

			self.rectifier = Predictor(device=device)
			self.rectifier.loadModel(rectifier_config)

		trans = [t for t in data_args['trans'] if not t.startswith('Tar_')]
		self.composer = transform.Composer(trans)

		self.confidence_table = None
		confidence_path = config.localPath('confidence.yaml')
		if os.path.exists(confidence_path):
			with open(confidence_path, 'r') as file:
				self.confidence_table = yaml.safe_load(file)
				logging.info('confidence_table loaded: %s', confidence_path)


	def concatImages (self, images):
		staves = []
		piece_segments = []

		for image in images:
			pieces = sliceFeature(image, width=self.slicing_width, overlapping = 2 / MARGIN_DIVIDER, padding=True)
			pieces = np.array(list(pieces), dtype=np.uint8)
			data, _ = self.composer(pieces, np.ones((1, 4, 4, 2)))

			staves.append(data)
			piece_segments.append(pieces.shape[0])

		stack = np.concatenate(staves, axis=0)

		return stack, piece_segments


	def predict (self, input_paths, positions=None):
		with_rectifier = self.rectifier and positions is not None

		try:
			staves = []
			for bi in range(0, len(input_paths), PATH_BATCH_SIZE):
				batch_input_paths = input_paths[bi:bi + PATH_BATCH_SIZE]
				images = list(map(loadImage, batch_input_paths))
				stack, piece_segments = self.concatImages(images)

				# run model
				outputs = []
				with torch.no_grad():
					for i in range(0, stack.shape[0], BATCH_SIZE):
						batch = stack[i:i + BATCH_SIZE]
						batch = torch.from_numpy(batch).to(self.device)

						output = self.model(batch).cpu()
						outputs.append(output)
				outputs = torch.cat(outputs, dim=0)

				# find contours and to score semantic
				layer = 0
				for seg in piece_segments:
					output = outputs[layer:layer + seg]
					layer += seg

					semantic = spliceOutputTensor(output)
					semantic = np.uint8(semantic * 255)

					ss = ScoreSemantic(semantic, self.labels, confidence_table=self.confidence_table)
					if not with_rectifier:
						yield ss.json()
					else:
						staves.append(ss.json())

			if with_rectifier:
				assert len(positions) == len(staves), f'positions-staves length mismatch: {len(positions)}-{len(staves)}'

				groups = {}
				for pos, staff in zip(positions, staves):
					staff['staff'] = pos['staff']
					staff['staffY'] = pos['staffY']

					system = pos['system']
					groups[system] = groups.get(system, ScoreSystem())
					groups[system].appendStaff(staff)

				for system in groups.values():
					batch, segments = system.batchize(self.semantics, self.n_seq_max)
					inputs = (batch[key] for key in ['semantic', 'staff', 'x', 'y1', 'y2', 'confidence'])
					values = torch.sigmoid(self.rectifier.model(*inputs).squeeze(2))
					values = values.masked_select(batch['mask']).cpu().tolist()

					system.assignValues(values, segments)
					system.cleanup()

				logging.info('staves: %s', len(staves))
				for staff in staves:
					yield staff
		except:
			logging.warn('ScoreSemanticProcessor error: %s', sys.exc_info()[1])
			traceback.print_tb(sys.exc_info()[2], file=sys.stderr)
