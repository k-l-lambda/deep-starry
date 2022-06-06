
import os
import json
from fs import open_fs
from zipfile import ZipFile, ZIP_STORED
from tqdm import tqdm
#import logging
import dill as pickle
import torch

from ..notation import VELOCITY_MAX



def vectorizeNotationFile (file, with_ci=False):
	notation = json.load(file)
	notes = notation['notes']

	data = {
		'time': torch.tensor([note['start'] for note in notes], dtype=torch.float32),
		'pitch': torch.tensor([note['pitch'] for note in notes], dtype=torch.long),
		'velocity': torch.tensor([note['velocity'] / VELOCITY_MAX for note in notes], dtype=torch.float32),
	}

	if with_ci:
		data['ci'] = torch.tensor([note['ci'] for note in notes], dtype=torch.long)

	return data


def preprocessDataset (source_dir, target_path):
	source = open_fs(source_dir)
	target = ZipFile(target_path, 'w', compression=ZIP_STORED)

	dir_list = [name for name in source.listdir('/') if source.isdir(name)]

	for dirname in tqdm(dir_list, desc='Preprocess groups'):
		criterion_file = open(os.path.join(source_dir, dirname, 'criterion.json'), 'r')
		criterion = vectorizeNotationFile(criterion_file)

		sample_index = 0
		samples = []
		while os.path.exists(os.path.join(source_dir, dirname, f'{sample_index}.json')):
			sample_file = open(os.path.join(source_dir, dirname, f'{sample_index}.json'), 'r')
			samples.append(vectorizeNotationFile(sample_file, with_ci=True))

			sample_index += 1

		target_filename = f'{dirname}.pkl'

		tensors = {
			'criterion': criterion,
			'samples': samples,
		}
		target.writestr(target_filename, pickle.dumps(tensors))

	target.close()
