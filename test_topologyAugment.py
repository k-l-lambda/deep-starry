
import sys
import math
import json
import numpy as np
from perlin_noise import PerlinNoise
from tqdm import tqdm

from starry.topology.data.semantics import loadClusterSet, distortElements



if __name__ == '__main__':
	with open(sys.argv[1], 'r') as file:
		data = loadClusterSet(file)
		clusters = data.get('clusters')

		for cluster in tqdm(clusters):
			elements = cluster['elements']

			noise = PerlinNoise(octaves=1/8)
			xfactor = math.exp(np.random.randn() * 0.3)
			positions = distortElements(elements, noise, xfactor)

			for elem, pos in zip(elements, positions):
				elem['x'] = pos['x']
				elem['y1'] = pos['y1']
				elem['y2'] = pos['y2']

		with open('./test.json', 'w') as out:
			json.dump(data, out)
