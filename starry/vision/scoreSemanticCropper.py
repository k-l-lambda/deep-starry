
import os
import numpy as np
from tqdm import tqdm
import cv2

from ..utils.dataset_factory import loadDataset
from .score_semantic import ScoreSemantic, VERTICAL_UNITS



STAMP_SIZE = 36
HALF_STAMP_SIZE = STAMP_SIZE // 2


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

		index = 0

		for feature, target in tqdm(self.data):
			#print('feature:', feature.shape, target.shape)

			unit_size = feature.shape[2] // VERTICAL_UNITS

			source = np.uint8(feature[0, 0] * 255)
			heatmap = np.uint8(target[0] * 255)
			semantic = ScoreSemantic(heatmap, self.labels)
			#print('semantic:', semantic.data)

			for point in semantic.data['points']:
				x, y = point['x'] * unit_size, point['y'] * unit_size + feature.shape[2] // 2
				ll, rr = int(x - HALF_STAMP_SIZE), int(x + HALF_STAMP_SIZE)
				tt, bb = int(y - HALF_STAMP_SIZE), int(y + HALF_STAMP_SIZE)
				l, r = max(0, ll), min(source.shape[1], rr)
				t, b = max(0, tt), min(source.shape[0], bb)
				if r > l and b > t:
					stamp = np.ones((STAMP_SIZE, STAMP_SIZE), dtype=np.uint8) * 255
					stamp[t - tt:b - tt, l - ll:r - ll] = source[t:b, l:r]

					stamp_path = os.path.join(outdir, point["semantic"], f'{index}-{int(x)},{int(y)}.png')
					cv2.imwrite(stamp_path, stamp)
					#print('point:', stamp_path)

			index += 1
