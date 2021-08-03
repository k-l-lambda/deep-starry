
import os
from datetime import date
import yaml

from .env import *



TRAINING_DIR = os.environ.get('TRAINING_DIR', 'training')


class Configuration:
	@staticmethod
	def create (file_path):
		filename = os.path.basename(file_path)
		filename = os.path.splitext(filename)[0]
		today = date.today().strftime('%Y%m%d')

		data = yaml.safe_load(open(file_path, 'r'))
		data['id'] = data['id'].format(filename=filename, date=today)

		dir = os.path.join(TRAINING_DIR, data['id'])
		os.makedirs(dir, exist_ok=True)

		return Configuration(dir, data)


	def __init__ (self, dir, data=None):
		self.dir = dir
		self.data = data

		if data is None:
			self.load()
		else:
			self.save()

		if self.data['env'] is not None:
			for key, value in self.data['env'].items():
				os.environ[key] = str(value)
				#print('env set:', key, value)


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