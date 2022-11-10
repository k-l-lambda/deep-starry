
import numpy as np
import cv2



INDEX_MAP_SIZE = 4096


def softMax (xs, temperature):
	exps = [np.exp(x * temperature) for x in xs]
	s = sum(exps)

	return [e / s for e in exps]


class Distorter:
	def __init__ (self, noise_path='./temp/perlin.npy'):
		noise_path = [noise_path] if type(noise_path) == str else noise_path

		self.perlin_maps = [np.load(path) for path in noise_path]
		self.perlin_dimension = self.perlin_maps[0].shape[1]

		self.index_x = np.tile(np.arange(INDEX_MAP_SIZE, dtype=np.float32)[None, :], (INDEX_MAP_SIZE, 1))
		self.index_y = np.tile(np.arange(INDEX_MAP_SIZE, dtype=np.float32)[:, None], (1, INDEX_MAP_SIZE))


	def compoundMaps (self, indices, wt):
		weights = [1] if len(self.perlin_maps) == 1 else softMax(np.random.randn(len(self.perlin_maps)), wt)

		result = np.zeros(self.perlin_maps[0][0].shape, dtype = np.float32)

		for level, maps in enumerate(self.perlin_maps):
			result += maps[indices[level]] * weights[level]

		return result


	def make_maps (self, shape, scale, intensity, wt):
		xi = [np.random.randint(len(maps)) for maps in self.perlin_maps]
		yi = [np.random.randint(len(maps)) for maps in self.perlin_maps]

		shrink = 1 / (max(shape[0], shape[1]) * scale)

		sy, sx = shape[0] * shrink, shape[1] * shrink
		biasx = np.random.random() * (1 - sx) if sx < 1 else 1 - sx
		biasy = np.random.random() * (1 - sy) if sy < 1 else 1 - sy

		nm_x = np.abs(self.index_x[:shape[0], :shape[1]] * shrink + biasx) * self.perlin_dimension
		nm_y = np.abs(self.index_y[:shape[0], :shape[1]] * shrink + biasy) * self.perlin_dimension

		noise_x = cv2.remap(self.compoundMaps(xi, wt), nm_x, nm_y, cv2.INTER_CUBIC) * intensity + self.index_x[:shape[0], :shape[1]]
		noise_y = cv2.remap(self.compoundMaps(yi, wt), nm_x, nm_y, cv2.INTER_CUBIC) * intensity + self.index_y[:shape[0], :shape[1]]

		return noise_x, noise_y


	def distort (self, source, mapx, mapy, borderMode=cv2.BORDER_REPLICATE):	# source: (height, width, channel)
		if source.shape[2] > 4:
			channels = source.shape[2]
			result = np.zeros((mapx.shape[0], mapx.shape[1], channels), dtype=source.dtype)
			for c in range(0, channels, 4):
				src = source[:, :, c:min(c + 4, channels)]
				result[:, :, c:min(c + 4, channels)] = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR, borderMode=borderMode).reshape((mapx.shape[0], mapx.shape[1], min(4, channels - c)))

			return result

		return cv2.remap(source, mapx, mapy, cv2.INTER_LINEAR, borderMode=borderMode)
