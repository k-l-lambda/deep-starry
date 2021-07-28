
import json
import numpy as np

from .semantic_element import JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES, JOINT_TARGET_SEMANTIC_ELEMENT_TYPES



def loadConnectionSet (file):
	return json.load(file)


def elementToVector (elem, d_word):
	return [elem['type'], elem['staff']] # TODO: position sinusoid embed


def exampleToTensors (example, n_seq_max, d_word):
	elements = example['elements']
	seqs = np.ones((len(elements), n_seq_max), dtype=np.float32)
	for i, elem in enumerate(elements):
		seq = elementToVector(elem, d_word)
		seqs[i, :len(seq)] = seq

	matrixH = np.zeros((n_seq_max, n_seq_max), dtype=np.float32)
	for i, line in enumerate(example['matrixH']):
		matrixH[i, :len(line)] = line

	masks = [[False] * n_seq_max for i in range(n_seq_max)]
	for i in range(len(elements)):
		if elements[i]['type'] in JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES:
			for j in range(len(elements)):
				if elements[j]['type'] in JOINT_TARGET_SEMANTIC_ELEMENT_TYPES:
					masks[i][j] = True

	return seqs, matrixH, masks
