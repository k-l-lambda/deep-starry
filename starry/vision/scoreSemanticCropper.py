
import os
import numpy as np
from tqdm import tqdm

from ..utils.dataset_factory import loadDataset
from .score_semantic import ScoreSemantic, VERTICAL_UNITS



class ScoreSemanticCropper:
	def __init__ (self, config, data_dir):
		self.labels = config['data.args.labels']

		config['data.splits'] = '0/1'
		config['data.batch_size'] = 1
		self.data, = loadDataset(config, data_dir=data_dir)


	def run (self, outdir):
		# make dirs
		for label in self.labels:
			os.makedirs(os.path.join(outdir, label), exist_ok=True)

		for feature, target in tqdm(self.data):
			print('feature:', feature.shape, target.shape)

			unit_size = feature.shape[2] // VERTICAL_UNITS

			heatmap = np.uint8(target[0] * 255)
			semantic = ScoreSemantic(heatmap, self.labels)
			print('semantic:', semantic.data)

			# TODO: save stamp images

			break
