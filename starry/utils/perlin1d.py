
import os
import numpy as np



PERLIN1D_CACHE_PATH = os.getenv('PERLIN1D_CACHE_PATH')


def interpolate (arr, x):
	assert x >= 0 and x <= len(arr) - 1

	xi = int(x)
	frac = x - xi
	y1, y2 = arr[xi], arr[xi + 1]

	return y1 * (1 - frac) + y2 * frac


class Perlin1d:
	def __init__ (self):
		self.cache = np.load(PERLIN1D_CACHE_PATH)


	def get (self, size, cycle):
		SCALES, N, RESOLUTION = self.cache.shape
		#print('cache:', SCALES, N, RESOLUTION)

		octave = size / cycle
		scale = int(max(0, np.ceil(np.log2(octave))))

		assert scale < SCALES, f'scale out of range: {scale} >= {SCALES}'

		row = self.cache[scale, np.random.randint(N)]

		k = (RESOLUTION >> scale) / cycle
		bias = (RESOLUTION - size * k) * np.random.rand()

		return np.array([interpolate(row, bias + x * k) for x in range(size)])


	def integral (self, size, cycle, amplitude=1):
		rate = self.get(size, cycle)

		result = np.zeros(size)
		y = 0
		for x in range(size):
			result[x] = y
			y += np.exp(rate[x] * amplitude)

		return result
