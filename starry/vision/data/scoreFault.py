
import re
from fs import open_fs
from tqdm import tqdm
import json
import yaml
from zipfile import ZipFile, ZIP_DEFLATED
import dill as pickle
import logging



def vectorizePoints (points):
	pass


def preprocessDataset (source_dir, target_path, name_id=re.compile(r'(.+)-\d+\.\w+\.\w+$'), n_seq_max=0x100, d_word=0x200):
	source = open_fs(source_dir)
	target = ZipFile(target_path, 'w', compression=ZIP_DEFLATED)

	file_list = list(filter(lambda name: source.isfile(name) and name_id.match(name) is not None, source.listdir('/')))

	identifier = lambda name: name_id.match(name).group(1)

	id_map = dict(map(lambda name: (name, identifier(name)), file_list))
	ids = list(set(id_map.values()))
	ids.sort()

	example_infos = []
	groups = []

	for id in tqdm(ids, desc='Preprocess groups'):
		groups.append(id)
		filenames = [name for name, id_ in id_map.items() if id_ == id]

		points = []
		staves = set()
		for filename in filenames:
			staff = re.match(r'.*-(\d+)\.\w+', filename).group(1)
			staff = int(staff)
			if staff in staves:
				continue
			staves.add(staff)

			with source.open(filename, 'r') as file:
				group = json.load(file)

				staffY = group['staffY']
				for point in group['points']:
					point['staff'] = staff
					point['y'] += staffY
					points.append(point)

		#print('group:', id, staves, len(points))
		tensors = vectorizePoints(points)
		target.writestr(f'{id}.pkl', pickle.dumps(tensors))

		length = sum(map(lambda t: t.nelement(), tensors)) * 4

		example_infos.append({
			'group': id,
			'length': length,	# in bytes
		})

	logging.info('Dumping index.')
	target.writestr('index.yaml', yaml.dump({
		'd_word': d_word,
		'n_seq_max': n_seq_max,
		'examples': example_infos,
		'groups': groups,
	}))

	target.close()
