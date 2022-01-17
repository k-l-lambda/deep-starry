
import re
from fs import open_fs
from tqdm import tqdm
import json
import yaml
from zipfile import ZipFile, ZIP_STORED
import dill as pickle
import numpy as np
import torch
from torch.utils.data import IterableDataset
import logging

from .score import parseFilterStr



SYSTEM_ID = re.compile(r'(.+)-\d+\.\w+\.\w+$')
SCORE_ID = re.compile(r'(.+)-\d+-\d+\.\w+\.\w+$')

SEMANTIC_MAX = 78
STAFF_MAX = 64


def reduceIntervalIndices (indices, n, tolerance):
	if n <= tolerance:
		return [0]

	sorted_indices = sorted(indices)
	interval_map = dict(map(lambda i: (sorted_indices[i], (n if i == len(sorted_indices) - 1 else sorted_indices[i + 1]) - sorted_indices[i]),
		range(len(sorted_indices))))

	left_indices = [i for i in indices if i > 0]
	while len(left_indices) > 0:
		index = left_indices.pop()
		keys = list(interval_map.keys())
		#print('keys:', keys)
		prev = keys[keys.index(index) - 1]
		interval_map[prev] += interval_map[index]
		if interval_map[prev] > tolerance:
			left_indices.append(index)
			break

		del interval_map[index]

	return [0, *left_indices]


def segmentIntervals (indices, n, n_segment):
	#indices = reduceIntervalIndices(indices, n, n_segment // 4)
	sorted_indices = sorted(indices)
	index_intervals = list(map(lambda i: (sorted_indices[i], (n if i == len(sorted_indices) - 1 else sorted_indices[i + 1]) - sorted_indices[i]),
		range(len(sorted_indices))))
	intervals = list(map(lambda ii: ii[1], index_intervals))

	def fetch (start):
		length = 0
		for i in range(start, len(intervals)):
			if length + intervals[i] > n_segment:
				return (min(filter(lambda ii: sum(intervals[ii:i]) < n_segment // 4, range(start + 1, i + 1))), length)
			length += intervals[i]

		return (None, length)

	segments = []
	i = 0
	while i is not None:
		next, length = fetch(i)
		segments.append((i, length))
		i = next if next is not None and next < len(intervals) else None

	# extend last segment
	if len(segments) > 1:
		i, length = segments[-1]
		while i > 0 and length + intervals[i - 1] < n_segment:
			i -= 1
			length += intervals[i]
		segments[-1] = (i, length)

	return list(map(lambda seg: (index_intervals[seg[0]][0], seg[1]), segments))


class ScoreFault (IterableDataset):
	@staticmethod
	def loadPackage (url, splits='*0/1', device='cpu', **kwargs):
		splits = splits.split(':')
		package = open_fs(url)
		index_file = package.open('index.yaml', 'r')
		index = yaml.load(index_file)

		def loadEntries (split):
			phases, cycle = parseFilterStr(split)

			ids = [id for i, id in enumerate(index['groups']) if i % cycle in phases]

			return [entry for entry in index['examples'] if entry['group'] in ids]

		return tuple(map(lambda split: ScoreFault(
			package, loadEntries(split), device, shuffle='*' in split, **kwargs,
		), splits))


	def load (root, args, splits, device='cpu'):
		url = f'zip://{root}' if root.endswith('.zip') else root
		return ScoreFault.loadPackage(url, splits, device, **args)


	def __init__ (self, package, entries, device, shuffle=False, n_seq_max=0x100, confidence_temperature=0, position_drift=0):
		self.package = package
		self.entries = entries
		self.shuffle = shuffle
		self.device = device

		#self.n_seq_max = n_seq_max
		self.confidence_temperature = confidence_temperature
		self.position_drift = position_drift

		# segment entries
		for entry in self.entries:
			entry['segments'] = segmentIntervals(entry['intervals'], entry['points'], n_seq_max)


	def __len__(self):
		return sum(len(entry['segments']) for entry in self.entries)


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.entries)
		else:
			torch.manual_seed(0)

		for entry in self.entries:
			with self.package.open(entry['filename'], 'rb') as file:
				tensors = pickle.load(file)
				for i, length in entry['segments']:
					seg = dict((key, tensor[i:i + length]) for key, tensor in tensors.items())
					yield seg


	def collateBatch (self, batch):
		keys = batch[0].keys()
		n_seq = max([len(example['semantic']) for example in batch])
		#print('collateBatch:', len(batch), n_seq, keys)

		result = {key: torch.zeros((len(batch), n_seq), dtype=batch[0][key].dtype) for key in keys}
		for key in keys:
			for i, example in enumerate(batch):
				value = example[key]
				result[key][i, :len(value)] = value

		# noise augment
		if self.confidence_temperature > 0:
			result['confidence'] = torch.exp(torch.randn(len(batch), n_seq) * self.confidence_temperature)
		if self.position_drift > 0:
			result['x'] += torch.randn(len(batch), n_seq) * self.position_drift
			result['y1'] += torch.randn(len(batch), n_seq) * self.position_drift
			result['y2'] += torch.randn(len(batch), n_seq) * self.position_drift

		result['mask'] = torch.ones((len(batch), n_seq), dtype=torch.float32)
		for i, example in enumerate(batch):
			result['mask'][i, len(example['semantic']):] = 0

		for key, tensor in result.items():
			result[key] = tensor.to(self.device)

		return result


def vectorizePoints (points, semantics):
	semantic = []
	staff = []
	x = []
	y1 = []
	y2 = []
	confidence = []
	value = []

	for point in points:
		semantic.append(semantics.index(point['semantic']))
		staff.append(point['staff'])
		x.append(point['x'])
		y1_, y2_ = (point['extension']['y1'], point['extension']['y2']) if ('extension' in point and 'y1' in point['extension']) else (point['y'], point['y'])
		y1.append(y1_)
		y2.append(y2_)
		confidence.append(point['confidence'])
		value.append(point['value'])

	return {
		'semantic': torch.tensor(semantic, dtype=torch.int32),
		'staff': torch.tensor(staff, dtype=torch.int32),
		'x': torch.tensor(x, dtype=torch.float32),
		'y1': torch.tensor(y1, dtype=torch.float32),
		'y2': torch.tensor(y2, dtype=torch.float32),
		'confidence': torch.tensor(confidence, dtype=torch.float32),
		'value': torch.tensor(value, dtype=torch.float32),
	}


def preprocessDataset (source_dir, target_path, semantics):
	semantics = ['_PAD', *semantics]

	source = open_fs(source_dir)
	target = ZipFile(target_path, 'w', compression=ZIP_STORED)

	file_list = list(filter(lambda name: source.isfile(name) and SCORE_ID.match(name) is not None, source.listdir('/')))

	identifier = lambda name: SYSTEM_ID.match(name).group(1)

	id_map = dict(map(lambda name: (name, identifier(name)), file_list))
	ids = list(set(id_map.values()))
	ids.sort()

	example_infos = []
	groups = set()

	for id in tqdm(ids, desc='Preprocess groups'):
		filenames = [name for name, id_ in id_map.items() if id_ == id]

		points = []
		staves = set()
		for filename in filenames:
			group_id = SCORE_ID.match(filename).group(1)
			groups.add(group_id)

			staff = re.match(r'.*-(\d+)\.\w+', filename).group(1)
			staff = int(staff)
			if staff in staves or staff >= STAFF_MAX:
				continue
			staves.add(staff)

			with source.open(filename, 'r') as file:
				graph = json.load(file)

				staffY = graph['staffY']
				for point in graph['points']:
					point['staff'] = staff
					point['y'] += staffY
					points.append(point)

		points.sort(key=lambda point: point['x'])
		intervals = list(map(lambda i: (i, points[i + 1]['x'] - points[i]['x']), range(len(points) - 1)))
		intervals.sort(key=lambda interval: -interval[1])
		interval_indices = list(map(lambda interval: interval[0], intervals))
		interval_indices = reduceIntervalIndices(interval_indices, len(points), 64)

		#segments = segmentIntervals(interval_indices, len(points), 256)
		#print('segments:', segments)
		#print('interval_indices:', list(map(lambda i: points[i + 1]['x'] - points[i]['x'], interval_indices)))

		#print('graph:', id, staves, len(points))
		tensors = vectorizePoints(points, semantics)
		target_name = f'{id}.pkl'
		target.writestr(target_name, pickle.dumps(tensors))

		length = sum(map(lambda t: t.nelement(), tensors.values())) * 4

		example_infos.append({
			'filename': target_name,
			'group': group_id,
			'length': length,	# in bytes
			'points': len(points),
			'intervals': interval_indices,
		})

	logging.info('Dumping index.')
	target.writestr('index.yaml', yaml.dump({
		'examples': example_infos,
		'groups': list(groups),
		'semantic_max': len(semantics),
	}))

	target.close()
