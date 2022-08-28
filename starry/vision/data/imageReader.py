
import os
import sys
import platform
import numpy as np
from fs import open_fs
from fs.copy import copy_file
import PIL.Image
import logging



# workaround fs path seperator issue
_S = (lambda path: path.replace(os.path.sep, '/')) if platform.system() == 'Windows' else (lambda p: p)


class ImageReader:
	def __init__(self, address):
		self.fs = open_fs(address)


	def readImage (self, path):
		try:
			with self.fs.open(_S(path), 'rb') as file:
				image = PIL.Image.open(file)

				return np.array(image)
		except:
			logging.warn('error to open image:', path, sys.exc_info()[0])
			return None


	def exists (self, path):
		return self.fs.exists(_S(path))


	def listFiles (self, dir):
		return filter(lambda name: self.fs.isfile(_S(os.path.join(dir, name))), self.fs.listdir(dir))


class CachedImageReader:
	def __init__ (self, source_address):
		cache_path = os.path.split(source_address)[-1]

		self.source_reader = ImageReader(source_address)
		self.cache_reader = ImageReader(f'mem://{cache_path}')


	def readImage (self, path):
		if not self.cache_reader.exists(path):
			self.cache(path)

		return self.cache_reader.readImage(path)


	def cache (self, path):
		try:
			path = _S(path)
			dir = _S(os.path.dirname(path))
			if not self.cache_reader.fs.exists(dir):
				self.cache_reader.fs.makedir(dir)

			copy_file(self.source_reader.fs, path, self.cache_reader.fs, path)
		except:
			logging.warn('error to cache file: %s, %s', path, sys.exc_info()[1])


	def writeImage (self, path, data):
		with self.source_reader.fs.open(_S(path), 'wb') as file:
			image = PIL.Image.fromarray(data)
			image.save(file, PIL.Image.registered_extensions()['.png'])

		self.cache(path)


	def exists (self, path):
		return self.source_reader.exists(path)


	def cached (self, path):
		return self.cache_reader.exists(path)


def makeReader (root):
	name, ext = os.path.splitext(root)
	is_zip = ext == '.zip'
	nomalized_root = name if is_zip else root
	reader_url = ('zip://' + root) if is_zip else root

	return ImageReader(reader_url), nomalized_root
