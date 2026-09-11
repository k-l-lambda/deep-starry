
import os
from torch.utils.data import DataLoader, IterableDataset

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

	def build_loader (dataset):
		# A MAP-STYLE dataset must have its SAMPLER shuffled here: DataLoader reaches it through
		# `__getitem__` and never calls its `__iter__`, so every map-style class in this repo -- each
		# of which sets `self.shuffle` from the split's '*' and then shuffles inside `__iter__` -- was
		# served in file order on every epoch, with the shuffle silently dead. The flag was still
		# load-bearing elsewhere (it also drives `random_crop`), which is why this never looked broken.
		#
		# An IterableDataset keeps its own `__iter__`, where that shuffle DOES run, and DataLoader
		# rejects `shuffle=True` for one outright -- hence the branch rather than a bare getattr.
		shuffle = False if isinstance(dataset, IterableDataset) else bool(getattr(dataset, 'shuffle', False))
		return DataLoader(dataset, batch_size=batch_size or config['data.batch_size'],
			collate_fn=dataset.collateBatch, shuffle=shuffle)

	return tuple(map(build_loader, datasets))
