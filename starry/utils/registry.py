'''Decorator-based registries for models and datasets (cf. trigoRL's
trigor/{models,data}/registry.py).

A class registers itself at definition time via `@register_model('Name')` /
`@register_dataset('Name')`; the factories (model_factory / dataset_factory) then
look the name up in MODELS / DATASETS. Registration is triggered by IMPORTING the
module that defines the class — so the factories import only the modules a config
declares (its `imports:` list), avoiding pulling in unrelated modalities' heavy
optional dependencies (cv2, primesieve, ...). When a config declares no `imports:`,
the factories fall back to importing every modality package (backward compatible).
'''

import importlib
import logging


MODELS = {}
DATASETS = {}
# Run-local ASSETS: things a run must materialize in its own config directory BEFORE the dataset and
# model factories run, and must then reload verbatim on resume rather than regenerate. A synthesized
# vocabulary is the motivating case — regenerating it from today's assets would silently reinterpret a
# checkpoint's embedding rows. Registered the same way as models/datasets, so a config's `imports:`
# already triggers registration; see Configuration.preprocess for the create/resume contract.
ASSETS = {}


def _make_register (registry, kind):
	def register (name=None):
		'''Decorator (or direct call) registering a class under `name`.

		Usable as `@register_x('Name')`, bare `@register_x` (name = class __name__),
		or `register_x('Name', cls)`. Returns the class unchanged.
		'''
		def _register (cls):
			key = name if isinstance(name, str) else cls.__name__
			if key in registry and registry[key] is not cls:
				logging.debug('%s registry: overriding %s', kind, key)
			registry[key] = cls
			return cls

		# bare decorator: @register_x   (name is actually the class)
		if isinstance(name, type):
			cls, name = name, None
			return _register(cls)
		# direct call with explicit class: register_x('Name', cls) is not used here,
		# but support register_x(cls) defensively
		return _register

	return register


register_model = _make_register(MODELS, 'model')
register_dataset = _make_register(DATASETS, 'dataset')
register_asset = _make_register(ASSETS, 'asset')


def import_modules (specs):
	'''Import each dotted module path in `specs` to trigger its decorators. An import
	that fails (missing optional dependency) is logged and skipped, never raised, so
	one modality's missing dep can't block another's registration.'''
	for spec in specs or []:
		try:
			importlib.import_module(spec)
		except Exception as e:
			logging.warning('import_modules: failed to import %s: %s', spec, e)


def import_package_submodules (package):
	'''Import a package and every submodule directly under it (one level), so classes
	defined across its files self-register. Each submodule is imported in isolation —
	a failure (missing optional dep) is logged and skipped. Used by the factories'
	fallback path when a config declares no explicit `imports:`.'''
	import pkgutil

	try:
		pkg = importlib.import_module(package)
	except Exception as e:
		logging.warning('import_package_submodules: failed to import package %s: %s', package, e)
		return
	# namespace packages (no __init__) still expose __path__ for discovery
	for info in pkgutil.iter_modules(getattr(pkg, '__path__', [])):
		import_modules([f'{package}.{info.name}'])
