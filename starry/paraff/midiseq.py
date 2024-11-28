
import os
import dill as pickle
import yaml
import logging
from tqdm import tqdm
import torch

from starry.paraff.data.paragraph import MeasureLibrary



TOKENS = yaml.safe_load(open('./assets/midiseqVocab.yaml', 'r'))
T2I = {t: i for i, t in enumerate(TOKENS)}


def packMidiseqYaml (source_path, target_path=None):
	source_base = os.path.splitext(source_path)[0]
	target_path = target_path or (source_base + '.pkl')

	midiseq = yaml.safe_load(open(source_path, 'r'))

	seqs = []
	scoreIndices = [0]

	for name in tqdm(midiseq.keys()):
		measures = midiseq[name]
		ids = [[T2I[t] for t in m.split(' ')] for m in measures]

		seqs += ids
		scoreIndices.append(len(ids))

	package = dict(seqs=seqs, scoreIndices=scoreIndices)

	logging.info('Writing pickle file: %s', target_path)
	pickle.dump(package, open(target_path, 'wb'))


def summaryMeasures (source_path, n_seq, encoder_config):
	target_path = os.path.splitext(source_path)[0] + '-measures.pt'

	with open(source_path, 'rb') as paraff_file:
		mlib = MeasureLibrary(paraff_file, n_seq)
		summaries = mlib.encodeMeasures(encoder_config)

		torch.save(summaries, target_path)
