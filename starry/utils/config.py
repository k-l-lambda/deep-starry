
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
		created = data is not None

		if not created:
			self.load()
		self.preprocess(created=created, volatile=volatile)

		if created and not volatile:
			self.save()
		self._resolve_unified_vocab()

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

		data_args = self.data.get('data', {}).get('args', {})
		if self.data.get('data', {}).get('type') != 'Seq2Seq2':
			return
		source_format = self._seq2_format(data_args.get('source_format', 'midiseq2'))
		target_format = self._seq2_format(data_args.get('target_format', 'midiseq2'))
		if source_format == target_format == 'lilylet':
			raise ValueError('Lilylet -> Lilylet is not supported; at least one side must be midiseq2')
		if source_format == target_format:
			return
		if self.data.get('model') is None:
			raise ValueError('mixed Seq2Seq2 configuration requires a model section')

		from ..midi.data.unifiedSeq2Tokenizer import (
			build_unified_vocab, load_unified_vocab, write_unified_vocab)

		name = self.data.get('_unified_vocab')
		if created:
			if volatile:
				raise ValueError('volatile mixed configuration cannot create its run-local vocabulary; '
					'create the run persistently first')
			name = 'unifiedSeq2Vocab.json'
			path = self.localPath(name)
			artifact = build_unified_vocab()
			write_unified_vocab(path, artifact)
			self.data['_unified_vocab'] = name
		else:
			if not isinstance(name, str) or not name or os.path.isabs(name) or os.path.dirname(name):
				raise ValueError('mixed run state has no valid run-local _unified_vocab reference')
			path = self.localPath(name)
			artifact = load_unified_vocab(path)

		model_args = self.data['model'].setdefault('args', {})
		configured_size = model_args.get('vocab_size')
		if configured_size is not None and int(configured_size) != artifact['vocab_size']:
			raise ValueError(f'unified vocab_size must be {artifact["vocab_size"]}, got {configured_size}')
		model_args['vocab_size'] = artifact['vocab_size']
		# Persist relocatable references. They are resolved only after save, and never written back.
		data_args['vocab_path'] = name
		model_args['vocab_path'] = name

	@staticmethod
	def _seq2_format (value):
		value = str(value).lower()
		value = 'midiseq2' if value == 'midi' else value
		if value not in ('midiseq2', 'lilylet'):
			raise ValueError(f'unsupported Seq2Seq2 format {value!r}')
		return value

	def _resolve_unified_vocab (self):
		name = self.data.get('_unified_vocab')
		if name is None:
			return
		path = self.localPath(name)
		self.data['data']['args']['vocab_path'] = path
		self.data['model']['args']['vocab_path'] = path


	def localPath (self, name):
		return os.path.join(self.dir, name)


	def load (self):
		state_file = open(self.localPath('.state.yaml'), 'r')
		assert state_file is not None, f'No .state.yaml file found in config directory: {self.dir}'

		self.data = yaml.safe_load(state_file)


	def save (self):
		has_old = os.path.exists(self.localPath('.state.yaml'))
		if has_old:
			os.rename(self.localPath('.state.yaml'), self.localPath('~state.yaml'))
		data = copy.deepcopy(self.data)
		name = data.get('_unified_vocab')
		if name is not None:
			data['data']['args']['vocab_path'] = name
			data['model']['args']['vocab_path'] = name
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
