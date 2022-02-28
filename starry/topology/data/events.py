
import json
from torch import float32
import yaml
import re
from fs import open_fs
from zipfile import ZipFile, ZIP_STORED
from tqdm import tqdm
import logging
import dill as pickle
import math
import numpy as np
import torch
from perlin_noise import PerlinNoise

from ..event_element import BeamType, StemDirection



SCORE_ID = re.compile(r'(.+)\.\d+\.\w+\.\w+$')


NOISE_Y_SIGMA = 0.12


def distortElements (elements, noise, xfactor):
	def distort (elem):
		x = elem['x']
		x *= xfactor
		x += noise([elem['x'], elem['y1'] / 100])

		dy1 = np.random.randn() * NOISE_Y_SIGMA
		dy2 = dy1 if elem['y2'] == elem['y1'] else np.random.randn() * NOISE_Y_SIGMA

		return {
			'x': x,
			'y1': elem['y1'] + dy1,
			'y2': elem['y2'] + dy2,
		}

	return [distort(elem) for elem in elements]


def exampleToTensorsAugment (cluster, n_augment):
	elements = cluster['elements']
	n_seq = len(elements)

	opv = lambda x: 0 if x is None else x

	elem_type = torch.tensor([elem['type'] for elem in elements], dtype=torch.int32)
	staff = torch.tensor([opv(elem['staff']) for elem in elements], dtype=torch.int32)
	tick = torch.tensor([elem['tick'] for elem in elements], dtype=torch.float32)
	division = torch.tensor([opv(elem['division']) for elem in elements], dtype=torch.int32)
	dots = torch.tensor([opv(elem['dots']) for elem in elements], dtype=torch.int32)
	beam = torch.tensor([BeamType[elem['beam']] if elem['beam'] else 0 for elem in elements], dtype=torch.int32)
	direction = torch.tensor([StemDirection[elem['stemDirection']] if elem['stemDirection'] else 0 for elem in elements], dtype=torch.int32)
	grace = torch.tensor([1 if elem['grace'] else 0 for elem in elements], dtype=torch.float32)
	timeWarped = torch.tensor([1 if elem['timeWarped'] else 0 for elem in elements], dtype=torch.float32)
	fullMeasure = torch.tensor([1 if elem['fullMeasure'] else 0 for elem in elements], dtype=torch.float32)
	confidence = torch.tensor([1 for elem in elements], dtype=torch.float32)

	matrixH = torch.tensor(cluster['matrixH'], dtype=float32).flatten()

	feature = torch.zeros((n_augment, n_seq, 15), dtype=torch.float32)
	# TODO

	x = torch.zeros((n_augment, n_seq), dtype=torch.float32)
	y1 = torch.zeros((n_augment, n_seq), dtype=torch.float32)
	y2 = torch.zeros((n_augment, n_seq), dtype=torch.float32)
	for i in range(n_augment):
		noise = PerlinNoise(octaves=1/8)
		xfactor = math.exp(np.random.randn() * 0.3)
		positions = distortElements(elements, noise, xfactor)

		x[i] = torch.tensor([elem['x'] for elem in positions], dtype=torch.float32)
		y1[i] = torch.tensor([elem['y1'] for elem in positions], dtype=torch.float32)
		y2[i] = torch.tensor([elem['y2'] for elem in positions], dtype=torch.float32)

	return (
		# source
		elem_type,		# (n_seq)
		staff,			# (n_seq)
		feature,		# (n_augment, n_seq, 15)
		x,				# (n_augment, n_seq)
		y1,				# (n_augment, n_seq)
		y2,				# (n_augment, n_seq)

		# targets
		tick,			# (n_seq)
		division,		# (n_seq)
		dots,			# (n_seq)
		beam,			# (n_seq)
		direction,		# (n_seq)
		grace,			# (n_seq)
		timeWarped,		# (n_seq)
		fullMeasure,	# (n_seq)
		confidence,		# (n_seq)
		matrixH,		# (n_seq * n_seq)
	)


def preprocessDataset (source_dir, target_path, n_augment=64):
	source = open_fs(source_dir)
	target = ZipFile(target_path, 'w', compression=ZIP_STORED)

	file_list = [name for name in source.listdir('/') if source.isfile(name)]

	identifier = lambda name: SCORE_ID.match(name).group(1)

	#id_map = dict(map(lambda name: (name, identifier(name)), file_list))
	id_map = {name: identifier(name) for name in file_list}
	ids = list(set(id_map.values()))
	ids.sort()

	#logging.info('ids: %s', '\n'.join(ids))

	example_infos = []

	for id in tqdm(ids, desc='Preprocess groups'):
		filenames = [name for name, id_ in id_map.items() if id_ == id]

		ci = 0
		for filename in filenames:
			with source.open(filename, 'r') as file:
				cluster_set = json.load(file)

				for cluster in cluster_set['clusters']:
					target_filename = f'{id}-{ci}.pkl'

					tensors = exampleToTensorsAugment(cluster, n_augment)
					target.writestr(target_filename, pickle.dumps(tensors))

					length = sum(map(lambda t: t.nelement(), tensors)) * 4

					example_infos.append({
						'filename': target_filename,
						'group': id,
						'length': length,	# in bytes
					})

					ci += 1

	logging.info('Dumping index.')
	target.writestr('index.yaml', yaml.dump({
		'examples': example_infos,
		'groups': ids,
	}))

	target.close()
