
import os
import yaml
import json
import re
import logging
import torch
import dill as pickle



SEMANTIC_TABLE = yaml.safe_load(open('./assets/timewiseSemantics.yaml', 'r'))

SEMANTIC_MAX = len(SEMANTIC_TABLE)

TG_PAD = SEMANTIC_TABLE.index('_PAD')
TG_EOS = SEMANTIC_TABLE.index('_EOS')

STAFF_MAX = 3


def vectorizeMeasure (measure, n_seq_max):
	left, right = measure['left'], measure['right']

	points = [dict(
		semantic=SEMANTIC_TABLE.index(p['semantic']),
		staff=p['staff'],
		x=p['x'],
		y=p['y'],
		sy1=p['sy1'],
		sy2=p['sy2'],
		confidence=p['confidence'],
		) for p in measure['points'] if p['semantic'] in SEMANTIC_TABLE and p['staff'] < STAFF_MAX]
	# EOS
	points.append({
		'semantic': SEMANTIC_TABLE.index('_EOS'),
		'staff': 0,
		'x': right,
		'y': 0,
		'sy1': 0,
		'sy2': 0,
		'confidence': 100,
	})
	points = sorted(points, key=lambda p: p['semantic'])[:n_seq_max]	# clip points prior by semantic
	points = sorted(points, key=lambda p: p['x'])

	semantic = torch.tensor([p['semantic'] for p in points], dtype=torch.uint8)
	staff = torch.tensor([p['staff'] for p in points], dtype=torch.uint8)
	x = torch.tensor([p['x'] - left for p in points], dtype=torch.float)
	y = torch.tensor([p['y'] for p in points], dtype=torch.float)
	sy1 = torch.tensor([p['sy1'] for p in points], dtype=torch.float16)
	sy2 = torch.tensor([p['sy2'] for p in points], dtype=torch.float16)
	confidence = torch.tensor([p['confidence'] for p in points], dtype=torch.float16)

	return semantic, staff, x, y, sy1, sy2, confidence


def preprocessGraph (paragraph_file, json_dir, n_seq_max=512):
	with open(paragraph_file, 'r') as pfile:
		meta = yaml.safe_load(pfile)

		target_path = os.path.join(os.path.dirname(paragraph_file), re.sub(r'\.paraff$', '-semantic.pkl', meta['paraff']))
		#print('target_path:', target_path)

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

		semantic, staff, x, y, sy1, sy2, confidence = [torch.zeros(n_measure, n_seq_max, dtype=dtype) for dtype in [torch.uint8, torch.uint8, torch.float, torch.float, torch.float16, torch.float16, torch.float16]]
		lib_tensors = [semantic, staff, x, y, sy1, sy2, confidence]

		graph = None
		n_seqs = []

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

			for i, measure in enumerate(measures):
				tensors = vectorizeMeasure(measure, n_seq_max=n_seq_max)
				n_seqs.append(tensors[0].shape[0])
				for it, t in enumerate(tensors):
					lib_tensors[it][range0 + i][:t.shape[0]] = t

			logging.info('%d measures wrote.', range1 - range0)

		n_seqs = torch.tensor(n_seqs).float()
		logging.info('max n_seq: %d', n_seqs.max())
		logging.info('mean n_seq: %f', n_seqs.mean())

		with open(target_path, 'wb') as taget_file:
			pickle.dump(dict(semantic=semantic, staff=staff, x=x, y=y, sy1=sy1, sy2=sy2, confidence=confidence), taget_file)

		logging.info('Package saved: %s', target_path)
