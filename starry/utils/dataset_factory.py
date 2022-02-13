
import os
from torch.utils.data import DataLoader



type_dict = None


def registerTypes ():
	global type_dict

	from ..vision.data import PerisData

	classes = [
		PerisData,
	]

	type_dict = dict([(c.__name__, c) for c in classes])



def loadDataset (config, data_dir='.', device='cpu', splits=None, batch_size=None):
	global type_dict
	if type_dict is None:
		registerTypes()

	data_type = config['data.type']

	if data_type not in type_dict:
		raise RuntimeError("Dataset type %s not found" % data_type)

	dataset_class = type_dict[data_type]

	root = os.path.join(data_dir, config['data.root'])
	datasets = dataset_class.load(root, config['data.args'], args_variant=config['data.args_variant'],
		splits=splits or config['data.splits'], device=device)
	loaders = tuple(map(
		lambda dataset:
			DataLoader(dataset, batch_size=batch_size or config['data.batch_size'], collate_fn=dataset.collateBatch),
		datasets))

	return loaders
