
import torch
import logging

from .registry import MODELS, import_modules, import_package_submodules


# Modality packages whose submodules' import triggers @register_model on every class
# they expose. Used as the fallback when a config declares no `imports:` (backward
# compatible). Each submodule is imported independently so one modality's missing
# optional dep (cv2, primesieve, ...) can't block the others.
_MODALITY_MODELS = [
	'starry.topology.models',
	'starry.vision.models',
	'starry.paraff.models',
	'starry.lilylet.models',
]


def _ensure_registered (imports=None):
	if imports:
		import_modules(imports)
	else:
		for pkg in _MODALITY_MODELS:
			import_package_submodules(pkg)


# Backward-compat shims for callers predating the registry refactor (e.g. convertToOnnx.py
# uses `registerModels()` then membership tests against `model_dict`). `model_dict` aliases
# the live MODELS dict, so it reflects registrations performed in place.
model_dict = MODELS


def registerModels ():
	_ensure_registered()


def loadModel (config, postfix='', imports=None):
	_ensure_registered(imports)

	model_type = config['type'] + postfix

	if model_type not in MODELS:
		raise RuntimeError("Model type %s not found" % model_type)

	model_class = MODELS[model_type]

	return model_class(**config['args'])


def loadModelAndWeights (config, checkpoint_name=None, device='cpu', postfix='', imports=None):
	model = loadModel(config['model'], postfix=postfix, imports=imports)

	checkpoint = {}
	if checkpoint_name is not None:
		checkpoint = torch.load(config.localPath(checkpoint_name), map_location=device)
		model.load_state_dict(checkpoint['model'])
		logging.info('Weights file loaded: %s', config.localPath(checkpoint_name))

	return model, checkpoint
