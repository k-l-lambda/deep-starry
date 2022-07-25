
import os
import json
import yaml
from fs import open_fs
from zipfile import ZipFile, ZIP_STORED
from tqdm import tqdm
#import logging
import dill as pickle
import numpy as np
import torch

from ..notation import VELOCITY_MAX, KEYBOARD_SIZE, KEYBOARD_BEGIN



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


def vectorizeRegularNotationFileToFrames (file):
	notation = json.load(file)
	notes = notation['notes']

	n_frames = notes[-1]['chi'] + 1

	chi = torch.arange(n_frames, dtype=torch.long)
	time = torch.zeros(n_frames, dtype=torch.float32)
	frame = torch.zeros((n_frames, KEYBOARD_SIZE), dtype=torch.float32)

	for chi in range(n_frames):
		cns = [note for note in notes if note['chi'] == chi]
		time[chi] = cns[0]['start']

		for note in cns:
			frame[chi][note['pitch'] - KEYBOARD_BEGIN] = note['velocity'] / VELOCITY_MAX

	return {
		'chi': chi,
		'time': time,
		'frame': frame,
	}


FRAME_MEAN_DURATION = 32

NOISE_NOTE_P = 0.01


def rollPitch ():
	p = round(KEYBOARD_SIZE // 2 + np.random.randn() * KEYBOARD_SIZE / 4)

	return max(min(p, KEYBOARD_SIZE - 1), 0)


def vectorizeNoizyNotationFileToFrames (file):
	notation = json.load(file)
	notes = notation['notes']

	t = notes[0]['start']

	time = []
	chis = []
	frames = []

	while len(notes) > 0:
		end_t = t + FRAME_MEAN_DURATION * np.exp(np.random.randn() * 0.4)
		frame = torch.zeros(KEYBOARD_SIZE, dtype=torch.float32)
		chi = None

		while len(notes) > 0 and notes[0]['start'] >= t and notes[0]['start'] < end_t:
			chi = notes[0]['chi'] if chi is None else min(chi, notes[0]['chi'])
			note = notes.pop(0)
			pitch = note['pitch'] - KEYBOARD_BEGIN
			if pitch >= 0 and pitch < KEYBOARD_SIZE:
				frame[pitch] = note['velocity'] / VELOCITY_MAX

		while np.random.rand() < NOISE_NOTE_P:
			chi = -1 if chi is None else chi
			frame[rollPitch()] = 1 - np.random.rand() ** 2

		if chi is not None:
			chis.append(chi)
			time.append(t)
			frames.append(frame)

		t = end_t

	return {
		'chi': torch.tensor(chis, dtype=torch.long),
		'time': torch.tensor(time, dtype=torch.float32),
		'frame': torch.stack(frames, dim=0),
	}


def preprocessDataset (source_dir, target_path):
	source = open_fs(source_dir)
	target = ZipFile(target_path, 'w', compression=ZIP_STORED)

	dir_list = [name for name in source.listdir('/') if source.isdir(name)]

	example_infos = []

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

		example_infos.append({
			'name': dirname,
			'criterion': {
				'length': criterion['time'].shape[0],
			},
			'samples': [{'length': sample['ci'].shape[0]} for sample in samples],
		})

	target.writestr('index.yaml', yaml.dump({
		'examples': example_infos,
	}))

	target.close()


def preprocessDatasetFrames (source_dir, target_path):
	source = open_fs(source_dir)
	target = ZipFile(target_path, 'w', compression=ZIP_STORED)

	dir_list = [name for name in source.listdir('/') if source.isdir(name)]

	example_infos = []

	for dirname in tqdm(dir_list, desc='Preprocess groups'):
		criterion_path = os.path.join(source_dir, dirname, 'criterion.json')
		if not os.path.isfile(criterion_path):
			continue
		criterion_file = open(criterion_path, 'r')
		criterion = vectorizeRegularNotationFileToFrames(criterion_file)

		sample_index = 0
		samples = []
		while os.path.exists(os.path.join(source_dir, dirname, f'{sample_index}.json')):
			sample_file = open(os.path.join(source_dir, dirname, f'{sample_index}.json'), 'r')
			samples.append(vectorizeNoizyNotationFileToFrames(sample_file))

			sample_index += 1

		target_filename = f'{dirname}.pkl'

		tensors = {
			'criterion': criterion,
			'samples': samples,
		}
		target.writestr(target_filename, pickle.dumps(tensors))

		example_infos.append({
			'name': dirname,
			'criterion': {
				'length': criterion['time'].shape[0],
			},
			'samples': [{'length': sample['chi'].shape[0]} for sample in samples],
		})

	target.writestr('index.yaml', yaml.dump({
		'examples': example_infos,
	}))

	target.close()
