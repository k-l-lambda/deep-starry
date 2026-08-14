
import copy
import os
from datetime import date
import yaml
import logging

from .env import *



TRAINING_DIR = os.environ.get('TRAINING_DIR', 'training')


class Configuration:
	@staticmethod
	def createOrLoad (config_path, volatile=False):
		return Configuration.create(config_path, volatile=volatile) if config_path.endswith('.yaml') else Configuration(config_path, volatile=volatile)


	@staticmethod
	def create (file_path, volatile=False):
		filename = os.path.basename(file_path)
		filename = os.path.splitext(filename)[0]
		if filename.endswith('.local'):
			filename = os.path.splitext(filename)[0]
		today = date.today().strftime('%Y%m%d')

		data = yaml.safe_load(open(file_path, 'r'))
		data['id'] = data['id'].format(filename=filename, date=today)

		dir = os.path.join(TRAINING_DIR, data['id'])
		if not volatile:
			os.makedirs(dir, exist_ok=True)

		return Configuration(dir, data, volatile=volatile)


	def __init__ (self, dir, data=None, volatile=False):
		self.dir = dir
		self.data = data
		# Set by _use_scratch_dir for a volatile config that declares assets; see there.
		self._scratch = None
		created = data is not None

		if not created:
			# load() resolves pinned asset references; preprocess() re-validates them against the
			# relative names, so the order below (load -> preprocess -> resolve) stays idempotent.
			self.load()
		self.preprocess(created=created, volatile=volatile)

		if created and not volatile:
			self.save()
		self._resolve_assets()

		if self['env'] is not None:
			self.setEnv(self['env'])


	@classmethod
	def setEnv (cls, env):
		for key, value in env.items():
			if os.environ.get(key) is None:
				os.environ[key] = str(value)
				logging.info('env set: %s=%s', key, value)


	def preprocess (self, created=False, volatile=False):
		copy_fileds = self.data.get('_copy_fileds')
		if copy_fileds is not None:
			for fields in copy_fileds:
				field_target, field_source = fields
				self[field_target] = self[field_source]
			self.data.pop('_copy_fileds')

		self._preprocess_assets(created=created, volatile=volatile)

	def _preprocess_assets (self, created=False, volatile=False):
		'''Materialize the run-local assets a config declares, or reload the ones it pinned.

		A config's top-level `assets:` is a list of `{type, args?}`, each naming a class in the ASSETS
		registry (populated by the same `imports:` that populate MODELS/DATASETS). The contract:

			create(config, args) -> reference   once, when the run directory is made
			resume(config, reference)          on every later load; MUST NOT regenerate
			resolve(config, reference)         relative stored reference -> absolute in-memory path
			relativize(data, reference)        the inverse, applied to a copy on its way to disk

		`_assets` in the state maps type name -> that reference, which is what makes a run directory
		relocatable. This loop knows nothing about any particular modality; anything vocabulary- or
		format-specific belongs in the builder or in the dataset class.
		'''
		specs = self.data.get('assets')
		if not specs:
			return
		if not isinstance(specs, list):
			raise ValueError("config 'assets' must be a list of {type, args} entries")

		from .registry import ASSETS

		# The builders live in modality packages, so the config's imports must have run first.
		self._import_asset_modules()
		references = self.data.get('_assets') if not created else {}
		if not created and not isinstance(references, dict):
			raise ValueError("run state declares 'assets' but has no '_assets' reference map")
		for spec in specs:
			if not isinstance(spec, dict) or not isinstance(spec.get('type'), str):
				raise ValueError(f'invalid assets entry {spec!r}; expected {{type, args}}')
			name = spec['type']
			if name not in ASSETS:
				raise ValueError(f'asset type {name!r} not found; declare its module in imports:. '
					f'registered: {sorted(ASSETS)}')
			builder = ASSETS[name]
			if created:
				if volatile:
					self._use_scratch_dir()
				references[name] = builder.create(self, spec.get('args') or {})
			else:
				if name not in references:
					raise ValueError(f'run state has no reference for declared asset {name!r}')
				builder.resume(self, references[name])
		if created:
			self.data['_assets'] = references

	def _use_scratch_dir (self):
		'''Point a VOLATILE run at a throwaway directory, so its assets can be published somewhere.

		`volatile=True` means "build this config in memory and touch no run directory" — the read-only
		mode that validation notebooks and the inference tools use (`Configuration.createOrLoad(...,
		volatile=True)`). Such a config still needs its assets to EXIST, because the feeder and the
		model read them by path while being constructed. Publishing into `TRAINING_DIR/<id>` is what
		volatile promises not to do: that directory belongs to a real run, which may already exist.

		So the assets go to a temporary directory instead. It is owned by this Configuration — the
		TemporaryDirectory object is kept alive as an attribute, so the files last exactly as long as
		anything can still read the paths derived from them, and are cleaned up when the config is
		collected. Only ever called on a volatile config that declares assets; every other config keeps
		the directory it was given.
		'''
		if getattr(self, '_scratch', None) is None:
			import tempfile

			self._scratch = tempfile.TemporaryDirectory(prefix='starry-volatile-assets-')
			self.dir = self._scratch.name

	def _import_asset_modules (self):
		'''Import the config's `imports:` so the ASSETS registry is populated.

		Called from every asset entry point rather than once, because `load()` resolves references
		before `preprocess()` runs — and `load()` is also called on its own mid-training by the
		distributed trainer. Importing is idempotent and cached by sys.modules, so repeating it is free.
		'''
		from .registry import import_modules

		import_modules(self.data.get('imports'))

	def _each_asset (self):
		'''(builder, reference) for every pinned asset. Empty for a config that declares none.

		Raises on an unregistered type rather than skipping it: `save()` relies on this to relativize
		every reference, so a silent skip would write ABSOLUTE paths into the state and quietly cost
		the run its relocatability.
		'''
		from .registry import ASSETS

		references = self.data.get('_assets') or {}
		if references:
			self._import_asset_modules()
		for name, reference in references.items():
			if name not in ASSETS:
				raise ValueError(f'run state pins asset {name!r}, which is not registered; '
					f'declare its module in imports:. registered: {sorted(ASSETS)}')
			yield ASSETS[name], reference

	def _resolve_assets (self):
		for builder, reference in self._each_asset():
			builder.resolve(self, reference)


	def localPath (self, name):
		return os.path.join(self.dir, name)


	def load (self):
		state_file = open(self.localPath('.state.yaml'), 'r')
		assert state_file is not None, f'No .state.yaml file found in config directory: {self.dir}'

		self.data = yaml.safe_load(state_file)
		# The state holds run-relative asset references; in memory they are always absolute paths. The
		# distributed trainer reloads mid-epoch (trainerQuantitative.py), so without this a pinned
		# path would silently degrade to a bare filename after the first save.
		self._resolve_assets()


	def save (self):
		has_old = os.path.exists(self.localPath('.state.yaml'))
		if has_old:
			os.rename(self.localPath('.state.yaml'), self.localPath('~state.yaml'))
		data = copy.deepcopy(self.data)
		for builder, reference in self._each_asset():
			builder.relativize(data, reference)
		with open(self.localPath('.state.yaml'), 'w') as state_file:
			yaml.dump(data, state_file)

		if has_old:
			os.remove(self.localPath('~state.yaml'))


	@property
	def id (self):
		return self.data['id']


	def __getitem__ (self, key_path):
		fields = key_path.split('.')
		item = self.data
		for field in fields:
			if item is None:
				break

			item = item.get(field)

		return item


	def __setitem__ (self, key_path, value):
		fields = key_path.split('.')
		item = self.data
		for field in fields[:-1]:
			item = item.setdefault(field, {})

		if item is not None:
			item[fields[-1]] = value
