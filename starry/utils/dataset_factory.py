
import os
from torch.utils.data import DataLoader

from .registry import DATASETS, import_modules, import_package_submodules


# Modality packages whose submodules' import triggers @register_dataset on every class
# they expose. Fallback when a config declares no `imports:` (backward compatible). Each
# submodule is imported independently so one modality's missing optional dep can't block
# the others.
_MODALITY_DATA = [
	'starry.vision.data',
	'starry.topology.data',
	'starry.paraff.data',
	'starry.lilylet.data',
	'starry.midi.data',
]


def _ensure_registered (imports=None):
	if imports:
		import_modules(imports)
	else:
		for pkg in _MODALITY_DATA:
			import_package_submodules(pkg)


def loadDataset (config, data_dir='.', device='cpu', splits=None, batch_size=None):
	_ensure_registered(config['imports'])

	data_type = config['data.type']

	if data_type not in DATASETS:
		raise RuntimeError("Dataset type %s not found" % data_type)

	dataset_class = DATASETS[data_type]

	root = os.path.join(data_dir, config['data.root'])
	datasets = dataset_class.load(root, config['data.args'], args_variant=config['data.args_variant'],
		splits=splits or config['data.splits'], device=device)
	loaders = tuple(map(
		lambda dataset:
			DataLoader(dataset, batch_size=batch_size or config['data.batch_size'], collate_fn=dataset.collateBatch),
		datasets))

	return loaders
