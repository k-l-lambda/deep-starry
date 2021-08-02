
import os
from datetime import date
import yaml

from .env import *



TRAINING_DIR = os.environ.get('TRAINING_DIR', 'training')


class Configuration:
	@staticmethod
	def create (file_path):
		filename = os.path.basename(file_path)
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


	def load (self):
		state_file = open(os.path.join(self.dir, '.state.yaml'), 'r')
		assert state_file is not None, f'No .state.yaml file found in config directory: {self.dir}'

		self.data = yaml.safe_load(state_file)


	def save (self):
		with open(os.path.join(self.dir, '.state.yaml'), 'w') as state_file:
			yaml.dump(self.data, state_file)
