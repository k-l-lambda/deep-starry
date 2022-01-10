
import json
import yaml
import numpy as np
import torch
import os
import platform
from fs import open_fs
from zipfile import ZipFile, ZIP_DEFLATED
import re
from tqdm import tqdm
import math
import logging
import dill as pickle
from perlin_noise import PerlinNoise

from .semantic_element import JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES, JOINT_TARGET_SEMANTIC_ELEMENT_TYPES, ROOT_NOTE_SEMANTIC_ELEMENT_TYPES, STAFF_MAX



def loadClusterSet (file):
	return json.load(file)


ANGLE_CYCLE = 1000	# should be comparable with (but larger than) value's up limit

def get_sinusoid_vec (x, d_hid):
	vec = np.array([x / np.power(ANGLE_CYCLE / 2 * np.pi, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)], dtype=np.float32)
	vec[0::2] = np.sin(vec[0::2])
	vec[1::2] = np.cos(vec[1::2])

	return vec


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

	return list(map(distort, elements))


def sinusoid_pos (x, y1, y2, d_word):
	d_pos = d_word // 2
	x_vec = get_sinusoid_vec(x, d_pos)
	y1_vec = get_sinusoid_vec(y1, d_pos)
	y2_vec = get_sinusoid_vec(y2, d_pos)

	return np.concatenate((x_vec, y1_vec + y2_vec))	# d_word


def element_token (elem):
	se_type = elem['type']
	staff = elem['staff']
	staff = STAFF_MAX + staff if staff < 0 else staff % STAFF_MAX

	return se_type, staff


def elementToVector (elem, d_word):
	pos_vec = sinusoid_pos(elem['x'], elem['y1'], elem['y2'], d_word)
	token = element_token(elem)

	return token, pos_vec


def exampleToTensors (example, n_seq_max, d_word, matrix_placeholder=False, pruned_maskv=False):
	elements = example['elements']

	seq_id = np.ones((n_seq_max, 2), dtype=np.int32)
	seq_position = np.zeros((n_seq_max, d_word), dtype=np.float32)
	for i, elem in enumerate(elements):
		if i >= n_seq_max:
			break
		ids, position = elementToVector(elem, d_word)
		seq_id[i, :] = ids
		seq_position[i, :] = position

	groupsV = example.get('groupsV')
	maskv = [
			i < len(elements) and elements[i]['type'] in ROOT_NOTE_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)
		] if pruned_maskv or groupsV is None else [
			any(map(lambda group: i in group, groupsV)) for i in range(n_seq_max)
		]

	masks = (
		[i < len(elements) and elements[i]['type'] in JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)],
		[i < len(elements) and elements[i]['type'] in JOINT_TARGET_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)],
		maskv,
	)

	if matrix_placeholder:
		matrixH = [[0]]
		matrixV = [[0]]
	else:
		matrixH = example['compactMatrixH']

		# matrixV
		if groupsV is not None:
			i2g = [next(g for g, group in enumerate(groupsV) if i in group) if masks[2][i] else -1 for i in range(n_seq_max)]
			matrixV = [
				[(1 if i2g[i] == i2g[j] else 0) for j, mj in enumerate(masks[2]) if mj]
					for i, mi in enumerate(masks[2]) if mi
			]
	matrixH = np.array(matrixH, dtype=np.float32).flatten()

	if matrixV is not None:
		matrixV = np.array(matrixV, dtype=np.float32)
		triu = np.triu(np.ones(matrixV.shape)) == 0
		matrixV = matrixV[triu]

	return (
		seq_id,			# (n_seq_max, 2)
		seq_position,	# (n_seq_max, d_word)
		masks,			# (3, n_seq_max)
		matrixH,		# n_source_joints * n_target_joints
		matrixV,		# n_grouped * n_grouped
	)


def exampleToTensorsAugment (example, n_seq_max, d_word, n_augment, matrix_placeholder=False, pruned_maskv=False):
	elements = example['elements']

	seq_id = np.ones((n_seq_max, 2), dtype=np.int32)
	for i, elem in enumerate(elements):
		if i >= n_seq_max:
			break
		ids = element_token(elem)
		seq_id[i, :] = ids

	seq_position = np.zeros((n_augment, n_seq_max, d_word), dtype=np.float32)
	for i in range(n_augment):
		noise = PerlinNoise(octaves=1/8)
		xfactor = math.exp(np.random.randn() * 0.3)
		positions = distortElements(elements, noise, xfactor)

		for j, pos in enumerate(positions[:n_seq_max]):
			seq_position[i, j, :] = sinusoid_pos(pos['x'], pos['y1'], pos['y2'], d_word)

	groupsV = example.get('groupsV')
	maskv = [
			i < len(elements) and elements[i]['type'] in ROOT_NOTE_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)
		] if pruned_maskv else [
			any(map(lambda group: i in group, groupsV)) for i in range(n_seq_max)
		]

	masks = (
		[i < len(elements) and elements[i]['type'] in JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)],
		[i < len(elements) and elements[i]['type'] in JOINT_TARGET_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)],
		maskv,
	)

	if matrix_placeholder:
		matrixH = [[0]]
		matrixV = [[0]]
	else:
		matrixH = example['compactMatrixH']

		# matrixV
		i2g = [next(g for g, group in enumerate(groupsV) if i in group) if masks[2][i] else -1 for i in range(n_seq_max)]
		matrixV = [
			[(1 if i2g[i] == i2g[j] else 0) for j, mj in enumerate(masks[2]) if mj]
				for i, mi in enumerate(masks[2]) if mi
		]
	matrixH = np.array(matrixH, dtype=np.float32).flatten()

	matrixV = np.array(matrixV, dtype=np.float32)
	triu = np.triu(np.ones(matrixV.shape)) == 0
	matrixV = matrixV[triu]

	# to torch tensors
	seq_id = torch.from_numpy(seq_id).unsqueeze(0)
	seq_position = torch.from_numpy(seq_position)
	masks = torch.tensor(list(masks)).unsqueeze(0).repeat(n_augment, 1, 1)
	matrixH = torch.from_numpy(matrixH).unsqueeze(0)
	matrixV = torch.from_numpy(matrixV).unsqueeze(0)

	return (
		seq_id,			# (1, n_seq_max, 2)
		seq_position,	# (n_augment, n_seq_max, d_word)
		masks,			# (1, 3, n_seq_max)
		matrixH,		# (1, n_source_joints * n_target_joints)
		matrixV,		# (1, n_grouped * n_grouped)
	)


def toFixedBatchTensor (batch):
	length = max(0, *[len(arr) for arr in batch])
	fixed = np.zeros((len(batch), length), dtype=np.float32)
	for i, arr in enumerate(batch):
		fixed[i, :len(arr)] = arr

	return torch.tensor(fixed)


def batchizeTensorExamples (examples, batch_size):
	batches = []

	for i in range(0, len(examples), batch_size):
		ex = examples[i:min(len(examples), i + batch_size)]

		seq_id = torch.tensor(list(map(lambda x: x[0], ex)))
		seq_position = torch.tensor(list(map(lambda x: x[1], ex)))
		masks = torch.tensor(list(map(lambda x: x[2], ex)))

		matrixHs = list(map(lambda x: x[3], ex))
		matrixHsFixed = toFixedBatchTensor(matrixHs)

		matrixVs = list(map(lambda x: x[4], ex))
		matrixVsFixed = toFixedBatchTensor(matrixVs)

		batches.append({
			'seq_id': seq_id,				# int32		(n, seq, 2)
			'seq_position': seq_position,	# float32	(n, seq, d_word)
			'mask': masks,					# bool		(n, 3, seq)
			'matrixH': matrixHsFixed,		# float32	(n, max_batch_matricesH)
			'matrixV': matrixVsFixed,		# float32	(n, max_batch_matricesV)
		})

	return batches


class Dataset:
	# '*' in split means shuffle
	@staticmethod
	def loadPackage (file, batch_size, splits='*1,2,3,4,5,6,7,8/10:9/10', device='cpu'):
		splits = splits.split(':')
		data = Dataset.joinDataFromPackage(file)

		def extractExamples (split):
			split = split[1:] if split[0] == '*' else split

			phases, cycle = split.split('/')
			phases = list(map(int, phases.split(',')))
			cycle = int(cycle)

			ids = [id for i, id in enumerate(data['ids']) if i % cycle in phases]
			logging.info(f'splitted ids: {", ".join(ids)}')

			return sum([examples for i, examples in enumerate(data['groups']) if i % cycle in phases], [])

		return tuple(map(
			lambda split: Dataset(extractExamples(split), batch_size, device, '*' in split),
			splits))


	@staticmethod
	def joinDataFromPackage (file):
		file.seek(0, 2)
		total_size = file.tell()
		file.seek(0, 0)

		data = {}
		while file.tell() < total_size:
			chunk = pickle.load(file)
			for key, value in chunk.items():
				if data.get(key) and type(data[key]) == list:
					data[key] += value
				else:
					data[key] = value

		return data


	def __init__ (self, examples, batch_size, device, shuffle=False):
		self.examples = examples
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.device = device

	def __len__(self):
		return math.ceil(len(self.examples) / self.batch_size)

	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.examples)

		for i in range(0, len(self.examples), self.batch_size):
			ex = self.examples[i:min(len(self.examples), i + self.batch_size)]

			seq_id = torch.tensor(list(map(lambda x: x[0], ex))).to(self.device)
			seq_position = torch.tensor(list(map(lambda x: x[1], ex))).to(self.device)
			masks = torch.tensor(list(map(lambda x: x[2], ex))).to(self.device)

			matrixHs = list(map(lambda x: x[3], ex))
			matrixHsFixed = toFixedBatchTensor(matrixHs).to(self.device)

			matrixVs = list(map(lambda x: x[4], ex))
			matrixVsFixed = toFixedBatchTensor(matrixVs).to(self.device)

			yield {
				'seq_id': seq_id,				# int32		(n, seq, 2)
				'seq_position': seq_position,	# float32	(n, seq, d_word)
				'mask': masks,					# bool		(n, 3, seq)
				'matrixH': matrixHsFixed,		# float32	(n, max_batch_matricesH)
				'matrixV': matrixVsFixed,		# float32	(n, max_batch_matricesV)
			}


class DatasetScatter:
	@staticmethod
	def loadPackage (url, batch_size, splits='*1,2,3,4,5,6,7,8/10:9/10', device='cpu'):
		splits = splits.split(':')
		package = open_fs(url)
		index_file = package.open('index.yaml', 'r')
		index = yaml.load(index_file)

		def loadEntries (split):
			split = split[1:] if split[0] == '*' else split

			phases, cycle = split.split('/')
			phases = list(map(int, phases.split(',')))
			cycle = int(cycle)

			ids = [id for i, id in enumerate(index['groups']) if i % cycle in phases]

			return [entry for entry in index['examples'] if entry['group'] in ids]

		return tuple(map(lambda split: DatasetScatter(
			package, loadEntries(split), batch_size, device, '*' in split,
		), splits))


	def __init__ (self, package, entries, batch_size, device, shuffle=False):
		self.package = package
		self.entries = entries
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.device = device


	def __len__(self):
		return len(self.entries)


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.entries)

		for entry in self.entries:
			with self.package.open(entry['filename'], 'rb') as file:
				tensors = pickle.load(file)
				seq_id, seq_position, masks, matrixH, matrixV = tensors

				n_sample = min(self.batch_size, seq_position.shape[0])
				samples = torch.multinomial(torch.ones(seq_position.shape[0]), n_sample)

				seq_id = seq_id.repeat(n_sample, 1, 1).to(self.device)
				seq_position = seq_position[samples].to(self.device)
				masks = masks.repeat(n_sample, 1, 1).to(self.device)
				matrixH = matrixH.repeat(n_sample, 1).to(self.device)
				matrixV = matrixV.repeat(n_sample, 1).to(self.device)

				yield {
					'seq_id': seq_id,				# int32		(n, seq, 2)
					'seq_position': seq_position,	# float32	(n, seq, d_word)
					'mask': masks,					# bool		(n, 3, seq)
					'matrixH': matrixH,				# float32	(n, max_batch_matricesH)
					'matrixV': matrixV,				# float32	(n, max_batch_matricesV)
				}


# workaround fs path seperator issue
_S = (lambda path: path.replace(os.path.sep, '/')) if platform.system() == 'Windows' else (lambda p: p)


def preprocessDataset (data_dir, output_file, name_id = re.compile(r'(.+)\.\w+$'), n_seq_max=0x100, d_word=0x200):
	fs = open_fs(data_dir)
	file_list = list(filter(lambda name: fs.isfile(name), fs.listdir('/')))
	#print('file_list:', file_list)

	identifier = lambda name: name_id.match(name).group(1)

	id_map = dict(map(lambda name: (name, identifier(name)), file_list))
	ids = list(set(id_map.values()))
	ids.sort()
	#id_indices = dict(zip(ids, range(len(ids))))
	#print('ids:', ids)

	pickle.dump({
		'd_word': d_word,
		'ids': ids,
	}, output_file)

	def dumpData (id):
		filenames = [name for name, id_ in id_map.items() if id_ == id]

		examples = []
		for filename in filenames:
			with fs.open(filename, 'r') as file:
				data = loadClusterSet(file)
				clusters = data.get('clusters')
				valid_clusters = list(filter(lambda cluster: any(map(lambda e: e['type'] in JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES, cluster['elements'])), clusters))

				if len(valid_clusters) < len(clusters):
					logging.warning(f'{filename} has empty cluster!')

				for i, cluster in enumerate(valid_clusters):
					cluster['id'] = f'{id}-{i}'

				examples += valid_clusters

		examples.sort(key=lambda x: x['id'])
		#ids = list(map(lambda ex: ex['id'], examples))
		#print('ids:', ids)
		examples = list(map(lambda ex: exampleToTensors(ex, n_seq_max, d_word), examples))

		pickle.dump({ 'groups': [examples] }, output_file)

	for id in tqdm(ids, desc='Preprocess groups'):
		dumpData(id)


def preprocessDatasetScatter (source_dir, target_path, name_id=re.compile(r'(.+)\.\w+$'), n_seq_max=0x100, d_word=0x200, n_augment=64):
	source = open_fs(source_dir)
	target = ZipFile(target_path, 'w', compression=ZIP_DEFLATED)

	file_list = list(filter(lambda name: source.isfile(name), source.listdir('/')))
	#logging.info('file_list: %s', file_list)

	identifier = lambda name: name_id.match(name).group(1)

	example_infos = []
	groups = []

	nl = int(math.log10(len(file_list))) + 1

	for i, filename in enumerate(file_list):
		logging.info(f'Processing: %0{nl}d/%d	%s', i, len(file_list), filename)

		with source.open(filename, 'r') as file:
			data = loadClusterSet(file)
			clusters = data.get('clusters')
			valid_clusters = list(filter(lambda cluster: any(map(lambda e: e['type'] in JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES, cluster['elements'])), clusters))

			group_id = identifier(filename)
			groups.append(group_id)

			for ci, cluster in enumerate(tqdm(valid_clusters, leave=False)):
				target_filename = f'{group_id}-{ci}.pkl'

				tensors = exampleToTensorsAugment(cluster, n_seq_max, d_word, n_augment)
				target.writestr(target_filename, pickle.dumps(tensors))

				id, pos, msk, H, V = tensors
				length = (sum(map(lambda t: t.nelement(), [id, msk, H, V])) + pos.nelement() // n_augment) * 4

				example_infos.append({
					'filename': target_filename,
					'group': group_id,
					'length': length,	# in bytes
				})

	logging.info('Dumping index.')
	target.writestr('index.yaml', yaml.dump({
		'd_word': d_word,
		'n_seq_max': n_seq_max,
		'n_augment': n_augment,
		'examples': example_infos,
		'groups': groups,
	}))

	target.close()
