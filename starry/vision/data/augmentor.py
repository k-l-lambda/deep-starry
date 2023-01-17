
import os
import numpy as np
import cv2
import math
import random

from .textureSet import TextureSet
from .distorter import Distorter



TEXTURE_SET_DIR = os.environ.get('TEXTURE_SET_DIR')
TEXTURE_SET_SIZE = int(os.environ.get('TEXTURE_SET_SIZE', 1))


def gaussianNoise (sigma, shape):
	return np.random.normal(0, sigma, shape).astype(np.float32)


def appendGaussianNoise (source, sigma):
	return source + gaussianNoise(np.abs(np.random.randn() * sigma), source.shape)


def roundn (x, n):
	return round(x / n) * n


class ScoreTinter:
	random_texture_iter = None
	regular_texture_iter = None

	def __init__ (self, texture_dir, frame_count, config, shuffle):
		if shuffle:
			if ScoreTinter.random_texture_iter is None:
				ScoreTinter.random_texture_iter = TextureSet(texture_dir).makeIterator(frame_size = (2048, 2048), frame_count = frame_count, shuffle = True)
			self.texture_iter = ScoreTinter.random_texture_iter
		else:
			if ScoreTinter.regular_texture_iter is None:
				ScoreTinter.regular_texture_iter = TextureSet(texture_dir).makeIterator(frame_size = (2048, 2048), frame_count = frame_count, shuffle = False)
			self.texture_iter = ScoreTinter.regular_texture_iter

		self.fore_scale_range = config['fore_scale_range']
		self.fore_blur_range = config['fore_blur_range']
		self.fore_sigma = config['fore_sigma']
		self.fore_pow = config['fore_pow']
		self.back_scale_range = config['back_scale_range']
		self.back_blur_range = config['back_blur_range']
		self.back_sigma = config['back_sigma']
		self.back_pow = config['back_pow']

	def tint (self, source):
		back_texture = self.texture_iter.get(source.shape[:2], scale_range = self.back_scale_range, blur_range = self.back_blur_range)
		back_texture = (back_texture / 255.).astype(np.float32) * 2 - 1.
		back_tex_intensity = np.random.randn() * self.back_sigma
		back_intensity = max(0.3, 1 - np.random.random() ** self.back_pow)

		fore_texture = self.texture_iter.get(source.shape[:2], scale_range = self.fore_scale_range, blur_range = self.fore_blur_range)
		fore_texture = (fore_texture / 255.).astype(np.float32) * 2 - 1.
		fore_tex_intensity = np.random.randn() * self.fore_sigma
		fore_intensity = max(0.2, 1 - np.random.random() ** self.fore_pow)

		composed = np.minimum(back_intensity + (back_texture - back_intensity) * back_tex_intensity, 1) + (source - 1) * (fore_intensity + fore_texture * fore_tex_intensity)

		return composed


class Augmentor:
	def __init__ (self, options, shuffle=True):
		self.tinter = None
		self.distorter = None
		self.gaussian_noise = 0
		self.affine = None
		self.crease = None
		self.gaussian_blur = 0
		self.flip_texture = None
		self.flip_intensity_range = None

		if options:
			self.skip_p = options.get('skip_p', 0)
			self.aa_scale = options.get('aa_scale', 1)	# anit-aliasing scale

			if options.get('tinter'):
				size = min(TEXTURE_SET_SIZE, 4) if not shuffle else TEXTURE_SET_SIZE
				self.tinter = ScoreTinter(TEXTURE_SET_DIR, size, options['tinter'], shuffle = shuffle)

			if options.get('distortion'):
				DISTORTION = options['distortion']
				noise_path = DISTORTION.get('noise', './temp/perlin.npy')
				self.distorter = Distorter(noise_path)
				self.distortion = {
					'scale':		DISTORTION.get('scale', 2),
					'scale_sigma':	DISTORTION.get('scale_sigma', 0.4),
					'intensity':	DISTORTION['intensity'],
					'intensity_sigma':	DISTORTION['intensity_sigma'],
					'noise_weights_sigma':	DISTORTION.get('noise_weights_sigma', 1),
					'squeeze_sigma': DISTORTION.get('squeeze_sigma', 0.2),
					'target_border': DISTORTION.get('target_border', 'BORDER_TRANSPARENT'),
				}

			if options.get('gaussian_noise'):
				self.gaussian_noise = options['gaussian_noise']['sigma']

			if options.get('affine'):
				AFFINE = options['affine']
				self.affine = {
					'padding_scale': AFFINE.get('padding_scale', 1),
					'padding_sigma': AFFINE.get('padding_sigma', 0.04),
					'angle_sigma': AFFINE['angle_sigma'],
					'scale_sigma': AFFINE['scale_sigma'],
					'scale_mu': AFFINE.get('scale_mu', 1),
					'scale_limit': AFFINE.get('scale_limit', float('inf')),
					'size_limit': AFFINE.get('size_limit', float('inf')),
					'size_fixed': AFFINE.get('size_fixed'),
					'std_size': AFFINE.get('std_size'),
					'round': AFFINE.get('round', 4),
				}

			if options.get('crease'):
				CREASE = options['crease']
				self.crease = {
					'intensity': CREASE.get('intensity', 1),
					'margin': CREASE.get('margin', 0.02),
					'width': CREASE.get('width', 0.08),
					'flip_p': CREASE.get('flip_p', 0.5),
					'outter_p': CREASE.get('outter_p', 0.5),
				}

			if options.get('flip_mark'):
				self.flip_intensity_range = options['flip_mark']['intensity_range']

			if options.get('gaussian_blur'):
				self.gaussian_blur = options['gaussian_blur']['sigma']

	def augment (self, source, target=None):
		w, h, _ = source.shape

		if self.skip_p > 0 and np.random.random() < self.skip_p:
			if self.affine and (self.affine['size_limit'] or self.affine['size_fixed']) or self.affine['scale_mu'] != 1:
				scale = min(math.exp(math.log(self.affine['scale_mu']) + np.random.randn() * self.affine['scale_sigma']), self.affine['scale_limit'])
				center = (source.shape[1] // 2, source.shape[0] // 2)
				mat = cv2.getRotationMatrix2D(center, 0, scale)
				rx, ry = np.random.random(), np.random.random()
				source = self.affineTransform(source, scale=scale, mat=mat, rx=rx, ry=ry)
				target = self.affineTransform(target, scale=scale, mat=mat, rx=rx, ry=ry)
				source = np.expand_dims(source, -1)
			return source, target

		if self.aa_scale > 1 and target is not None:
			target = cv2.resize(target, (target.shape[1] * self.aa_scale, target.shape[0] * self.aa_scale), interpolation=cv2.INTER_CUBIC)

		if self.flip_intensity_range is not None:
			origin = source
			if self.flip_texture is not None:
				texture = cv2.flip(self.flip_texture, 1)
				texture = cv2.resize(texture, source.shape[:2][::-1])
				texture = np.expand_dims(texture, -1)
				intensity = random.random() * (self.flip_intensity_range[1] - self.flip_intensity_range[0]) + self.flip_intensity_range[0]
				source = np.minimum(source, texture * intensity + (1 - intensity))
				#print('intensity:', intensity)

			self.flip_texture = origin

		if self.tinter:
			source = self.tinter.tint(source)

		# book crease
		if self.crease:
			intensity = 1 - np.random.random() ** self.crease['intensity']
			width = source.shape[1] * np.exp(np.random.randn() * 1) * self.crease['width']
			outter = np.random.random() < self.crease['outter_p']
			margin = source.shape[1] * np.exp(np.random.randn() * 1) * self.crease['margin'] if outter else 0
			bending = np.exp(np.random.randn() * 0.6) * 0.6
			bar = np.arange(source.shape[1], dtype=np.float)
			bar = (bar - margin) / width
			if outter:
				bar = -bar
			bar[bar <= 0] = 1
			bar = bar ** bending
			bar = np.clip(bar, 0, 1) * intensity + (1 - intensity)

			if np.random.random() < self.crease['flip_p']:
				bar = bar[::-1]

			source[:, :, 0] = source[:, :, 0] * bar

		if self.affine:
			std_scale = self.affine['std_size'] / max(w, h) if self.affine['std_size'] is not None else 1
			scale = std_scale * min(math.exp(math.log(self.affine['scale_mu']) + np.random.randn() * self.affine['scale_sigma']), self.affine['scale_limit'])
			angle = np.random.randn() * self.affine['angle_sigma']

			padding_scale = self.affine['padding_scale']
			padding_sigma = self.affine['padding_sigma']
			padding_scale_x = padding_scale * math.exp(np.random.randn() * padding_sigma)
			padding_scale_y = padding_scale * math.exp(np.random.randn() * padding_sigma)
			padding_y = max(round((padding_scale_y - 1) * source.shape[0] / 2), 0)
			padding_x = max(round((padding_scale_x - 1) * source.shape[1] / 2), 0)

			center = ((source.shape[1] * padding_scale) // 2, (source.shape[0] * padding_scale) // 2)
			mat = cv2.getRotationMatrix2D(center, angle, 1)	# getRotationMatrix2D scaling is a bug
			mat[0][0] *= scale
			mat[0][1] *= scale
			mat[1][0] *= scale
			mat[1][1] *= scale
			#print('target.1:', target.shape, scale, center, angle)

			rx, ry = np.random.random(), np.random.random()

			source = self.affineTransform(source, (padding_y, padding_x), mat, scale, rx, ry, cv2.BORDER_REFLECT_101)
			target = self.affineTransform(target, (padding_y, padding_x), mat, scale, rx, ry, cv2.BORDER_REPLICATE, aa_scale=self.aa_scale) if target is not None else target
			#print('target.2:', target.shape, (padding_y, padding_x))

			source = np.expand_dims(source, -1)

		if self.distorter:
			scale = self.distortion['scale'] * math.exp(np.random.randn() * self.distortion['scale_sigma'])
			intensity = self.distortion['intensity'] * math.exp(np.random.randn() * self.distortion['intensity_sigma'])
			nx, ny = self.distorter.make_maps(source.shape, scale, intensity, self.distortion['noise_weights_sigma'], squeeze=self.distortion['squeeze_sigma'])

			#source = self.distorter.distort(source, nx, ny, borderMode=cv2.BORDER_REFLECT_101)
			source = self.distorter.distort(source, nx, ny, borderMode=cv2.BORDER_REPLICATE)

			if target is not None:
				if self.aa_scale > 1:
					nx = cv2.resize(nx, (nx.shape[1] * self.aa_scale, nx.shape[0] * self.aa_scale), interpolation=cv2.INTER_LINEAR) * self.aa_scale
					ny = cv2.resize(ny, (ny.shape[1] * self.aa_scale, ny.shape[0] * self.aa_scale), interpolation=cv2.INTER_LINEAR) * self.aa_scale
				target = self.distorter.distort(target, nx, ny, borderMode=getattr(cv2, self.distortion['target_border']))

			if len(source.shape) < 3:
				source = np.expand_dims(source, -1)

		if self.gaussian_blur > 0:
			kernel = round(abs(np.random.randn() * self.gaussian_blur)) * 2 + 1
			#print('kernel:', kernel)
			if kernel > 1:
				source = cv2.GaussianBlur(source, (kernel, kernel), 0)
				source = np.expand_dims(source, -1)

		if self.gaussian_noise > 0:
			source = appendGaussianNoise(source, self.gaussian_noise)

		if self.aa_scale > 1 and target is not None:
			target = cv2.resize(target, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)

		return np.clip(source, 0, 1), target


	def affineTransform (self, image, padding=(0, 0), mat=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), scale=1, rx=0, ry=0, borderType=cv2.BORDER_REPLICATE, aa_scale=1):
		image = cv2.copyMakeBorder(image, padding[0] * aa_scale, padding[0] * aa_scale, padding[1] * aa_scale, padding[1] * aa_scale, borderType)

		size_limit = self.affine['size_limit'] * aa_scale
		size_fixed = self.affine['size_fixed']
		scaled_shape = (size_fixed * aa_scale, size_fixed * aa_scale) if size_fixed is not None else (min(image.shape[1] * scale, size_limit), min(image.shape[0] * scale, size_limit))
		scaled_shape = (roundn(scaled_shape[0], self.affine['round'] * aa_scale), roundn(scaled_shape[1], self.affine['round'] * aa_scale))

		rest_x, rest_y = math.floor(image.shape[1] * scale - scaled_shape[0]), math.floor(image.shape[0] * scale - scaled_shape[1])
		if rest_x != 0:
			mat[0][2] = rx * -rest_x
		if rest_y != 0:
			mat[1][2] = ry * -rest_y
		#print('mat:', rest_x, rest_y, mat)

		image = cv2.warpAffine(image, mat, scaled_shape, borderMode=borderType, flags=cv2.INTER_CUBIC)

		return image
