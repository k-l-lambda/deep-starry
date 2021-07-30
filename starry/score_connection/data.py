
import json
import numpy as np
import torch
import os
import platform
from fs import open_fs
import re
from tqdm import tqdm
import math

from .semantic_element import JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES, JOINT_TARGET_SEMANTIC_ELEMENT_TYPES, STAFF_MAX



def loadConnectionSet (file):
	return json.load(file)


ANGLE_CYCLE = 1000	# should be comparable with (but larger than) value's up limit

def get_sinusoid_vec(x, d_hid):
	vec = np.array([x / np.power(ANGLE_CYCLE / 2 * np.pi, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)], dtype=np.float32)
	vec[0::2] = np.sin(vec[0::2])
	vec[1::2] = np.cos(vec[1::2])

	return vec


def elementToVector (elem, d_word):
	d_pos = d_word // 2
	x_vec = get_sinusoid_vec(elem['x'], d_pos)
	y1_vec = get_sinusoid_vec(elem['y1'], d_pos)
	y2_vec = get_sinusoid_vec(elem['y2'], d_pos)

	pos_vec = np.concatenate((x_vec, y1_vec + y2_vec))	# d_word

	se_type = elem['type']
	staff = elem['staff']
	staff = STAFF_MAX + staff if staff < 0 else staff

	return (se_type, staff), pos_vec


def exampleToTensors (example, n_seq_max, d_word):
	elements = example['elements']

	seq_id = np.ones((n_seq_max, 2), dtype=np.int32)
	seq_position = np.zeros((n_seq_max, d_word), dtype=np.float32)
	for i, elem in enumerate(elements):
		ids, position = elementToVector(elem, d_word)
		seq_id[i, :] = ids
		seq_position[i, :] = position

	masks = (
		[i < len(elements) and elements[i]['type'] in JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)],
		[i < len(elements) and elements[i]['type'] in JOINT_TARGET_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)],
	)

	matrixH = [
		[x for j, x in enumerate(line) if masks[1][j]]
			for i, line in enumerate(example['matrixH']) if masks[0][i]
	]
	matrixH = np.array(matrixH, dtype=np.float32).flatten()

	return (
		seq_id,			# (n_seq_max, 2)
		seq_position,	# (n_seq_max, d_word)
		matrixH,		# n_source_joints * n_target_joints
		masks,			# (2, n_seq_max)
	)


def batchizeTensorExamples (examples, batch_size):
	batches = []

	for i in range(0, len(examples), batch_size):
		ex = examples[i:min(len(examples), i + batch_size)]

		seq_id = torch.tensor(list(map(lambda x: x[0], ex)))
		seq_position = torch.tensor(list(map(lambda x: x[1], ex)))
		masks = torch.tensor(list(map(lambda x: x[3], ex)))

		matrixHs = list(map(lambda x: x[2], ex))
		matrixLen = max(*[len(mtx) for mtx in matrixHs])
		matrixHsFixed = np.zeros((len(matrixHs), matrixLen), dtype=np.float32)
		for i, mtx in enumerate(matrixHs):
			matrixHsFixed[i, :len(mtx)] = mtx
		matrixHsFixed = torch.tensor(matrixHsFixed)

		batches.append({
			'seq_id': seq_id,				# int32		(n, seq, 2)
			'seq_position': seq_position,	# float32	(n, seq, d_word)
			'mask': masks,					# bool		(n, 2, seq)
			'matrixH': matrixHsFixed,		# float32	(n, max_batch_matrices)
		})

	return batches


class Dataset:
	@staticmethod
	def loadPackage (data, batch_size, shuffles=(True, False), device='cpu'):
		return tuple(map(
			lambda examples_shuffle: Dataset(examples_shuffle[0], batch_size, device, examples_shuffle[1]),
			zip(data['sets'], shuffles)))


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
			masks = torch.tensor(list(map(lambda x: x[3], ex))).to(self.device)

			matrixHs = list(map(lambda x: x[2], ex))
			matrixLen = max(0, *[len(mtx) for mtx in matrixHs])
			matrixHsFixed = np.zeros((len(matrixHs), matrixLen), dtype=np.float32)
			for i, mtx in enumerate(matrixHs):
				matrixHsFixed[i, :len(mtx)] = mtx
			matrixHsFixed = torch.tensor(matrixHsFixed).to(self.device)

			yield {
				'seq_id': seq_id,				# int32		(n, seq, 2)
				'seq_position': seq_position,	# float32	(n, seq, d_word)
				'mask': masks,					# bool		(n, 2, seq)
				'matrixH': matrixHsFixed,		# float32	(n, max_batch_matrices)
			}


# workaround fs path seperator issue
_S = (lambda path: path.replace(os.path.sep, '/')) if platform.system() == 'Windows' else (lambda p: p)


def preprocessDataset (data_dir, splits = ('1,2,3,4,5,6,7,8/10', '9/10'), name_id = re.compile(r'(.+)\.\w+$'),
	n_seq_max=0x100, d_word=0x200):
	fs = open_fs(data_dir)
	file_list = list(filter(lambda name: fs.isfile(name), fs.listdir('/')))
	#print('file_list:', file_list)

	identifier = lambda name: name_id.match(name).group(1)

	id_map = dict(map(lambda name: (identifier(name), name), file_list))
	ids = list(id_map.keys())
	ids.sort()
	id_indices = dict(zip(ids, range(len(ids))))

	def loadData (split):
		phases, cycle = split.split('/')
		phases = list(map(int, phases.split(',')))
		cycle = int(cycle)

		filenames = [name for id, name in id_map.items() if id_indices[id] % cycle in phases]

		examples = []
		#for filename in tqdm(filenames, desc='Preprocess files'):
		for filename in filenames:
			with fs.open(filename, 'r') as file:
				data = loadConnectionSet(file)
				examples += data['connections']

		examples = list(map(lambda ex: exampleToTensors(ex, n_seq_max, d_word), tqdm(examples, desc='Preprocess examples', mininterval=1.)))

		#return batchizeTensorExamples(examples, batch_size)
		return examples

	return {
		'sets': tuple(map(loadData, splits)),
		'd_word': d_word,
	}
