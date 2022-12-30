
from tqdm import tqdm

from ..utils.dataset_factory import loadDataset
from .score_semantic import ScoreSemantic



class ScoreSemanticCropper:
	def __init__ (self, config, data_dir):
		self.labels = config['data.args.labels']

		config['data.splits'] = '0/1'
		self.data, = loadDataset(config, data_dir=data_dir)


	def run (self, outdir):
		for feature, target in tqdm(self.data):
			print('feature:', feature.shape, target.shape)
			break
