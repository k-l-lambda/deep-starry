
import re
from fs import open_fs
from tqdm import tqdm
import json
import yaml
from zipfile import ZipFile, ZIP_DEFLATED
import dill as pickle
import logging



system_id = re.compile(r'(.+)-\d+\.\w+\.\w+$')
score_id = re.compile(r'(.+)-\d+-\d+\.\w+\.\w+$')


def vectorizePoints (points):
	pass


def preprocessDataset (source_dir, target_path, n_seq_max=0x100, d_word=0x200):
	source = open_fs(source_dir)
	target = ZipFile(target_path, 'w', compression=ZIP_DEFLATED)

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
		#tensors = vectorizePoints(points)
		#target.writestr(f'{id}.pkl', pickle.dumps(tensors))

		length = 0#sum(map(lambda t: t.nelement(), tensors)) * 4

		example_infos.append({
			'group': group_id,
			'length': length,	# in bytes
		})

	logging.info('Dumping index.')
	target.writestr('index.yaml', yaml.dump({
		'd_word': d_word,
		'n_seq_max': n_seq_max,
		'examples': example_infos,
		'groups': list(groups),
	}))

	target.close()
