
import os
import yaml
import json
import re
import logging
import torch



SEMANTIC_TABLE = yaml.safe_load(open('./assets/timewiseSemantics.yaml', 'r'))


def preprocessGraph (paragraph_file, json_dir, n_seq=512):
	with open(paragraph_file, 'r') as pfile:
		meta = yaml.safe_load(pfile)

		semantic_files = []
		for root, dirs, files in os.walk(json_dir):
			semantic_files += files

		group_to_semantic = dict()
		for group in meta['groups']:
			#logging.info(group)

			captures = re.match('^\d+\.', group)
			if captures is not None:
				prefix = captures[0]
			elif group.endswith('.spartito.json'):
				prefix = group[:-14]
			else:
				prefix = group

			filename = next(file for file in files if file.startswith(prefix))
			if filename is None:
				logging.error('Cannot find semantic file for %', group)
			else:
				group_to_semantic[group] = os.path.join(json_dir, filename)
		#print('group_to_semantic:', group_to_semantic)

		n_measure = meta['paragraphs'][-1]['sentenceRange'][1]

		semantic, staff, x, y, sy1, sy2, confidence = [torch.zeros(n_measure, n_seq, dtype=dtype) for dtype in [torch.uint8, torch.uint8, torch.float, torch.float, torch.float16, torch.float16, torch.float16]]

		graph = None

		for paragraph in meta['paragraphs']:
			group = paragraph['group']
			if graph is None or graph['group'] != group:
				graph = json.load(open(group_to_semantic[group], 'r'))
				graph['group'] = group
				logging.info('Group loaded: %s', group)

			range0, range1 = paragraph['sentenceRange']

			measures = graph['measures']
			mrange_captures = re.match(r'.*\[(\d+)-(\d+)\]$', paragraph['name'])
			if mrange_captures is not None:
				mbegin, mend = int(mrange_captures[1]), int(mrange_captures[2])
				measures = [m for m in graph['measures'] if m['measureIndex'] >= mbegin and m['measureIndex'] <= mend]
			assert len(measures) == range1 - range0, 'measure number mismatch: %s, %d, [%d:%d]' % (paragraph['name'], len(measures), range0, range1)

			# TODO:

			logging.info('%d measures wrote.', range1 - range0)
