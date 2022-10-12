
from tqdm import tqdm
import sys
import numpy as np
from perlin_noise import PerlinNoise



def gen (n, scales=6, resolution=1024, output_path='./perlin1d.npy'):
	buffer = np.zeros((scales, n, resolution), dtype=np.float32)

	for scale in tqdm(range(scales)):
		for i in range(n):
			noise = PerlinNoise(octaves=2**scale, seed=i + scale * n)
			offset = np.random.rand()
			buffer[scale, i] = [noise(ii / resolution + offset) for ii in range(resolution)]

	with open(output_path, 'wb') as file:
		np.save(file, buffer)

	print('Done.')


if __name__ == '__main__':
	gen(int(sys.argv[1]))
