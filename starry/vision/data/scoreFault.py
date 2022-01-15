
import re
from fs import open_fs
from tqdm import tqdm
import json
import yaml
from zipfile import ZipFile, ZIP_STORED
import dill as pickle
import torch
import logging



system_id = re.compile(r'(.+)-\d+\.\w+\.\w+$')
score_id = re.compile(r'(.+)-\d+-\d+\.\w+\.\w+$')


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

	file_list = list(filter(lambda name: source.isfile(name) and score_id.match(name) is not None, source.listdir('/')))

	identifier = lambda name: system_id.match(name).group(1)

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
			group_id = score_id.match(filename).group(1)
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

		#print('graph:', id, staves, len(points))
		tensors = vectorizePoints(points, semantics)
		target_name = f'{id}.pkl'
		target.writestr(target_name, pickle.dumps(tensors))

		length = sum(map(lambda t: t.nelement(), tensors.values())) * 4

		example_infos.append({
			'filename': target_name,
			'group': group_id,
			'length': length,	# in bytes
		})

	logging.info('Dumping index.')
	target.writestr('index.yaml', yaml.dump({
		'examples': example_infos,
		'groups': list(groups),
	}))

	target.close()
