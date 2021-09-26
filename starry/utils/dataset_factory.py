
import os
from torch.utils.data import DataLoader



type_dict = None


def registerTypes ():
	global type_dict

	from ..vision.data import RenderScore, ScoreMask, ScoreGauge, ScorePage

	classes = [
		RenderScore,
		ScoreMask,
		ScoreGauge,
		ScorePage,
	]

	type_dict = dict([(c.__name__, c) for c in classes])



def loadDataset (config, data_dir='.', device='cpu'):
	global type_dict
	if type_dict is None:
		registerTypes()

	data_type = config['data.type']

	if data_type not in type_dict:
		raise RuntimeError("Dataset type %s not found" % data_type)

	dataset_class = type_dict[data_type]

	root = os.path.join(data_dir, config['data.root'])
	datasets = dataset_class.load(root, config['data.args'], splits=config['data.splits'], device=device)
	loaders = tuple(map(
		lambda dataset:
			DataLoader(dataset, batch_size=config['data.batch_size'], collate_fn=dataset.collateBatch),
		datasets))

	return loaders
