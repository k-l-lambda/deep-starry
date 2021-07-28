
import json
import numpy as np
import torch

from .semantic_element import JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES, JOINT_TARGET_SEMANTIC_ELEMENT_TYPES, STAFF_MAX



def loadConnectionSet (file):
	return json.load(file)


ANGLE_CYCLE = 1000	# should be comparable with (but larger than) value's up limit

def get_sinusoid_vec(x, d_hid):
	vec = np.array([x / np.power(ANGLE_CYCLE * 2 * np.pi, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)], dtype=np.float32)
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

	seq_id = np.ones((n_seq_max, 2), dtype=np.int16)
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


def batchizeTensorExamples (examples, batch_size, device = 'cpu'):
	batches = []

	for i in range(0, len(examples), batch_size):
		ex = examples[i:min(len(examples), i + batch_size)]

		seq_id = torch.tensor(list(map(lambda x: x[0], ex))).to(device)
		seq_position = torch.tensor(list(map(lambda x: x[1], ex))).to(device)
		masks = torch.tensor(list(map(lambda x: x[3], ex))).to(device)

		matrixHs = list(map(lambda x: x[2], ex))
		matrixLen = max(*[len(mtx) for mtx in matrixHs])
		matrixHsFixed = np.zeros((len(matrixHs), matrixLen), dtype=np.float32)
		for i, mtx in enumerate(matrixHs):
			matrixHsFixed[i, :len(mtx)] = mtx
		matrixHsFixed = torch.tensor(matrixHsFixed).to(device)

		batches.append({
			'seq_id': seq_id,
			'seq_position': seq_position,
			'mask': masks,
			'matrixH': matrixHsFixed,
		})

	return batches
