
import json
import numpy as np
import torch

from .semantic_element import JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES, JOINT_TARGET_SEMANTIC_ELEMENT_TYPES



def loadConnectionSet (file):
	return json.load(file)


def elementToVector (elem, d_word):
	return [elem['type'], elem['staff']] # TODO: position sinusoid embed


def exampleToTensors (example, n_seq_max, d_word):
	elements = example['elements']
	seq = np.ones((n_seq_max, d_word + 2), dtype=np.float32)
	for i, elem in enumerate(elements):
		word = elementToVector(elem, d_word)
		seq[i, :len(word)] = word

	masks = (
		[i < len(elements) and elements[i]['type'] in JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)],
		[i < len(elements) and elements[i]['type'] in JOINT_TARGET_SEMANTIC_ELEMENT_TYPES for i in range(n_seq_max)],
	)

	matrixH = [
		[x for j, x in enumerate(line) if masks[1][j]]
			for i, line in enumerate(example['matrixH']) if masks[0][i]
	]
	matrixH = np.array(matrixH, dtype=np.float32).flatten()

	return (seq,	# (n_seq_max, d_word + 2)
		matrixH,	# n_source_joints * n_target_joints
		masks)		# (2, n_seq_max)


def batchizeTensorExamples (examples, batch_size, device = 'cpu'):
	batches = []

	for i in range(0, len(examples), batch_size):
		ex = examples[i:min(len(examples), i + batch_size)]

		seqs = torch.tensor(list(map(lambda x: x[0], ex))).to(device)
		masks = torch.tensor(list(map(lambda x: x[2], ex))).to(device)

		matrixHs = list(map(lambda x: x[1], ex))
		matrixLen = max(*[len(mtx) for mtx in matrixHs])
		matrixHsFixed = np.zeros((len(matrixHs), matrixLen), dtype=np.float32)
		for i, mtx in enumerate(matrixHs):
			matrixHsFixed[i, :len(mtx)] = mtx
		matrixHsFixed = torch.tensor(matrixHsFixed).to(device)

		batches.append({
			'seq': seqs,
			'mask': masks,
			'matrixH': matrixHsFixed,
		})

	return batches
