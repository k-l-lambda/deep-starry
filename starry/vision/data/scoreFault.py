
import re
from fs import open_fs
from tqdm import tqdm
import json
import yaml
from zipfile import ZipFile, ZIP_STORED
import dill as pickle
import numpy as np
import torch
import logging



SYSTEM_ID = re.compile(r'(.+)-\d+\.\w+\.\w+$')
SCORE_ID = re.compile(r'(.+)-\d+-\d+\.\w+\.\w+$')


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


class ScoreFault:
	@staticmethod
	def loadPackage (url, batch_size, splits='*1,2,3,4,5,6,7,8/10:9/10', device='cpu', **kwargs):
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

		return tuple(map(lambda split: ScoreFault(
			package, loadEntries(split), batch_size, device, shuffle='*' in split, **kwargs,
		), splits))


	def __init__ (self, package, entries, batch_size, device, shuffle=False, n_seq_max=0x100, confidence_temperature=0, position_drift=0):
		self.package = package
		self.entries = entries
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.device = device

		self.n_seq_max = n_seq_max
		self.confidence_temperature = confidence_temperature
		self.position_drift = position_drift


	def __len__(self):
		return len(self.entries)


	def __iter__ (self):
		if self.shuffle:
			np.random.shuffle(self.entries)

		for entry in self.entries:
			with self.package.open(entry['filename'], 'rb') as file:
				tensors = pickle.load(file)
				'''seq_id, seq_position, masks, matrixH, matrixV = tensors

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
				}'''


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
			if staff in staves:
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
	}))

	target.close()
