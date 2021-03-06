
import json
import yaml
import re
from fs import open_fs
from zipfile import ZipFile, ZIP_STORED
from tqdm import tqdm
import logging
import dill as pickle
import numpy as np
import torch
from perlin_noise import PerlinNoise

from ..event_element import FEATURE_DIM, EventElementType, BeamType, StemDirection, STAFF_MAX



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


def boolRandn (value, sigma=0.2, true_bias=0.6, false_bias=-2):
	return np.exp(np.random.randn() * sigma + (true_bias if value else false_bias))


def genElementFeature (elem, drop_source=False):
	feature = elem.get('feature')
	if feature is None or drop_source:
		if not elem['type'] in [EventElementType.CHORD, EventElementType.REST]:
			return torch.zeros(FEATURE_DIM, dtype=torch.float32)

		feature = {}

		feature['divisions'] = [0] * 7
		if elem['type'] == EventElementType.CHORD:
			feature['divisions'][0] = boolRandn(elem['division'] == 0)
			feature['divisions'][1] = boolRandn(elem['division'] == 1)
			feature['divisions'][2] = boolRandn(elem['division'] >= 2)
			feature['divisions'][3] = boolRandn(elem['division'] >= 3)
			feature['divisions'][4] = boolRandn(elem['division'] >= 4) * (1 if elem['division'] >= 4 else feature['divisions'][3])
			feature['divisions'][5] = boolRandn(elem['division'] >= 5) * (1 if elem['division'] >= 5 else feature['divisions'][4])
			feature['divisions'][6] = boolRandn(elem['division'] >= 6) * (1 if elem['division'] >= 6 else feature['divisions'][5])

			#feature['divisions'][3:] = sorted(feature['divisions'][3:], reverse=True)
		elif elem['type'] == EventElementType.REST:
			feature['divisions'] = [boolRandn(elem['division'] == d, false_bias=-3) for d in range(7)]

		dots = elem.get('dots', 0)
		feature['dots'] = [boolRandn(dots >= 1), boolRandn(dots >= 2, false_bias=-4)]

		may_beam = elem['type'] == EventElementType.CHORD and elem['division'] >= 3
		beam_false_bias = -2 if may_beam else -6

		feature['beams'] = [
			boolRandn(elem['beam'] == "Open", 0.1, false_bias=beam_false_bias),
			boolRandn(elem['beam'] == "Continue", 0.1, false_bias=beam_false_bias),
			boolRandn(elem['beam'] == "Close", 0.1, false_bias=beam_false_bias),
		]
		feature['stemDirections'] = [
			boolRandn(elem['stemDirection'] == "u", true_bias=1, false_bias=-8),
			boolRandn(elem['stemDirection'] == "d", true_bias=1, false_bias=-8),
		]

		feature['grace'] = boolRandn(elem['grace'], true_bias=0)

	return torch.tensor([
		*feature['divisions'], *feature['dots'], *feature['beams'], *feature['stemDirections'], feature['grace'],
	], dtype=torch.float32)


def clusterFeatureToTensors (clusters):
	batch_size = len(clusters)
	n_seq = max(len(cluster['elements']) for cluster in clusters)

	zeros = lambda: torch.zeros(batch_size, n_seq)

	elem_type = zeros().long()
	staff = zeros().long()
	x = zeros()
	y1 = zeros()
	y2 = zeros()
	feature = torch.zeros(batch_size, n_seq, FEATURE_DIM)

	opv = lambda x, default=0: default if x is None else x

	for i, cluster in enumerate(clusters):
		elements = cluster['elements']
		n_elem = len(elements)
		elem_type[i, :n_elem] = torch.tensor([elem['type'] for elem in elements], dtype=torch.long)
		staff[i, :n_elem] = torch.tensor([opv(elem['staff'], STAFF_MAX - 1) for elem in elements], dtype=torch.long)
		x[i, :n_elem] = torch.tensor([elem['x'] for elem in elements], dtype=torch.float32)
		y1[i, :n_elem] = torch.tensor([elem['y1'] for elem in elements], dtype=torch.float32)
		y2[i, :n_elem] = torch.tensor([elem['y2'] for elem in elements], dtype=torch.float32)

		for ei, elem in enumerate(elements):
			feature[i, ei, :] = genElementFeature(elem)

	return dict(
		type=elem_type,
		staff=staff,
		x=x,
		y1=y1,
		y2=y2,
		feature=feature,
	)


def exampleToTensorsAugment (cluster, n_augment):
	elements = cluster['elements']
	n_seq = len(elements)

	opv = lambda x, default=0: default if x is None else x

	elem_type = torch.tensor([elem['type'] for elem in elements], dtype=torch.int32)
	staff = torch.tensor([opv(elem['staff'], STAFF_MAX - 1) for elem in elements], dtype=torch.int32)
	tick = torch.tensor([elem['tick'] for elem in elements], dtype=torch.float32)
	division = torch.tensor([opv(elem['division']) for elem in elements], dtype=torch.int32)
	dots = torch.tensor([opv(elem['dots']) for elem in elements], dtype=torch.int32)
	beam = torch.tensor([BeamType[elem['beam']] if elem['beam'] else 0 for elem in elements], dtype=torch.int32)
	direction = torch.tensor([StemDirection[elem['stemDirection']] if elem['stemDirection'] else 0 for elem in elements], dtype=torch.int32)
	grace = torch.tensor([1 if elem['grace'] else 0 for elem in elements], dtype=torch.float32)
	timeWarped = torch.tensor([1 if elem['timeWarped'] else 0 for elem in elements], dtype=torch.float32)
	fullMeasure = torch.tensor([1 if elem['fullMeasure'] else 0 for elem in elements], dtype=torch.float32)
	fake = torch.tensor([elem.get('fake', 0) for elem in elements], dtype=torch.float32)

	rawMatrixH = torch.tensor(cluster['matrixH'], dtype=torch.float32)
	matrixH = rawMatrixH[1:, :-1]	# exlude BOS & EOS
	matrixH = matrixH.flatten()

	# relative tick mask
	tickSrc = tick.unsqueeze(-1).repeat(1, n_seq)
	tickTar = tick.unsqueeze(0).repeat(n_seq, 1)
	tickDiff = tickSrc - tickTar
	maskT = (rawMatrixH > 0) | torch.tril(tickDiff == 0, diagonal=-1)

	feature = torch.zeros((n_augment, n_seq, 15), dtype=torch.float32)
	x = torch.zeros((n_augment, n_seq), dtype=torch.float32)
	y1 = torch.zeros((n_augment, n_seq), dtype=torch.float32)
	y2 = torch.zeros((n_augment, n_seq), dtype=torch.float32)
	for i in range(n_augment):
		noise = PerlinNoise(octaves=1/8)
		xfactor = np.exp(np.random.randn() * 0.2)
		positions = distortElements(elements, noise, xfactor)

		x[i] = torch.tensor([elem['x'] for elem in positions], dtype=torch.float32)
		y1[i] = torch.tensor([elem['y1'] for elem in positions], dtype=torch.float32)
		y2[i] = torch.tensor([elem['y2'] for elem in positions], dtype=torch.float32)

		#stability = np.random.power(max(np.random.poisson(stability_base), 2))
		for j, elem in enumerate(elements):
			feature[i, j] = genElementFeature(elem, drop_source=((i + 1) % 4) == 0)

	return {
		# source
		'type':				elem_type,		# (n_seq)
		'staff':			staff,			# (n_seq)
		'feature':			feature,		# (n_augment, n_seq, 15)
		'x':				x,				# (n_augment, n_seq)
		'y1':				y1,				# (n_augment, n_seq)
		'y2':				y2,				# (n_augment, n_seq)

		# targets
		'tick':				tick,			# (n_seq)
		'division':			division,		# (n_seq)
		'dots':				dots,			# (n_seq)
		'beam':				beam,			# (n_seq)
		'stemDirection':	direction,		# (n_seq)
		'grace':			grace,			# (n_seq)
		'timeWarped':		timeWarped,		# (n_seq)
		'fullMeasure':		fullMeasure,	# (n_seq)
		'fake':				fake,			# (n_seq)
		'matrixH':			matrixH,		# ((n_seq - 1) * (n_seq - 1))
		'tickDiff':			tickDiff,		# (n_seq * n_seq)
		'maskT':			maskT,			# (n_seq * n_seq)
	}


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

					length = sum(map(lambda t: t.nelement(), tensors.values())) * 4

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
