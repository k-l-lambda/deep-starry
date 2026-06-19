
import os
import dill as pickle
import yaml
import logging
from tqdm import tqdm
import torch

from starry.paraff.data.paragraph import MeasureLibrary
from .midiseq import T2I



def packMidiseqYaml (source_path, target_path=None):
	source_base = os.path.splitext(source_path)[0]
	target_path = target_path or (source_base + '.pkl')

	midiseq = yaml.safe_load(open(source_path, 'r'))

	seqs = []

	scoreIdx = 0
	scoreIndices = [scoreIdx]

	n_seq_max = 0
	n_id = 0

	for name in tqdm(midiseq.keys()):
		measures = midiseq[name]
		ids = [[T2I[t] for t in m.split(' ')] for m in measures]

		lens = [len(s) for s in ids]
		n_seq_max = max(n_seq_max, *lens)
		n_id += sum(lens)

		seqs += ids
		scoreIdx += len(ids)
		scoreIndices.append(scoreIdx)

	package = dict(seqs=seqs, scoreIndices=scoreIndices)

	logging.info('Writing pickle file: %s', target_path)
	pickle.dump(package, open(target_path, 'wb'))

	logging.info('number of scores: %d', len(scoreIndices) - 1)
	logging.info('max sentence length: %d', n_seq_max)
	logging.info('average sentence length: %d', n_id / len(seqs))


def summaryMeasures (source_path, n_seq, encoder_config):
	target_path = os.path.splitext(source_path)[0] + '-measures.pt'

	with open(source_path, 'rb') as paraff_file:
		mlib = MeasureLibrary(paraff_file, n_seq)
		summaries = mlib.encodeMeasures(encoder_config)

		torch.save(summaries, target_path)
