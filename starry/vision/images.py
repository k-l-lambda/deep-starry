
import os
import PIL.Image
import cv2
import numpy as np



def arrayFromImageStream (stream):
	image = PIL.Image.open(stream)

	arr = np.array(image)
	if len(arr.shape) >= 3:
		return arr[:, :, :3]

	arr = np.expand_dims(arr, -1)

	return arr


def rm (filepath):
	if os.path.exists(filepath):
		os.unlink(filepath)


def writeImageFileFormat (data, path_format, i, type):
	filepath = path_format % {'i': i, 'type': type}
	rm(filepath)
	cv2.imwrite(filepath, data)
