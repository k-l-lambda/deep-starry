
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

		if data is None:
			self.load()
			self.preprocess()
		else:
			self.preprocess()

			if not volatile:
				self.save()

		if self['env'] is not None:
			self.setEnv(self['env'])


	@classmethod
	def setEnv (cls, env):
		for key, value in env.items():
			if os.environ.get(key) is None:
				os.environ[key] = str(value)
				logging.info('env set: %s=%s', key, value)


	def preprocess (self):
		copy_fileds = self.data.get('_copy_fileds')
		if copy_fileds is not None:
			for fields in copy_fileds:
				field_target, field_source = fields
				self[field_target] = self[field_source]
			self.data.pop('_copy_fileds')


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
		with open(self.localPath('.state.yaml'), 'w') as state_file:
			yaml.dump(self.data, state_file)

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
