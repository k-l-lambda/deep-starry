
import os
import io
import base64
import math
import PIL.Image
import cv2
import numpy as np
import random



MARGIN_DIVIDER = 8


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


def sliceFeature (source, width, overlapping=0.25, padding=False):	# source: (height, width, channel)
	step = math.floor(width - source.shape[0] * overlapping)

	if padding:
		margin = math.floor(source.shape[0] * overlapping) // 2
		center = source
		source = np.zeros((source.shape[0], source.shape[1] + margin * 2, source.shape[2]))
		source[:, margin:-margin, :] = center
		source[:, :margin, :] = center[:, :1, :]
		source[:, -margin:, :] = center[:, -1:, :]

	for x in range(0, source.shape[1], step):
		sliced_source = None

		if x + width <= source.shape[1]:
			sliced_source = source[:, x:x + width, :]
		else:
			sliced_source = np.ones((source.shape[0], width, source.shape[2]), dtype=np.float32) * 255
			sliced_source[:, :source.shape[1] - x, :] = source[:, x:, :]

		yield sliced_source


def splicePieces (pieces, magin_divider, keep_margin=False):	# pieces: (batch, channel, height, width)
	piece_height, piece_width = pieces.shape[2:]
	margin_width = piece_height // magin_divider
	patch_width = piece_width - margin_width * 2
	result = np.zeros((pieces.shape[1], pieces.shape[2], patch_width * pieces.shape[0]), dtype=np.float32)

	for i, piece in enumerate(pieces):
		result[:, :, i * patch_width : (i + 1) * patch_width] = piece[:, :, margin_width:-margin_width]

	if keep_margin:
		return np.concatenate((pieces[0, :, :, :margin_width], result, pieces[-1, :, :, -margin_width:]), axis = 2)

	return result	# (channel, height, width)


def softSplicePieces (pieces, magin_divider):
	batches, channels, piece_height, piece_width = pieces.shape
	overlap_width = piece_height * 2 // magin_divider
	segment_width = piece_width - overlap_width

	slope = np.arange(overlap_width, dtype = np.float32) / overlap_width
	slope = slope.reshape((1, 1, overlap_width))
	inv_slope = 1 - slope

	result = np.zeros((channels, piece_height, segment_width * batches + overlap_width), dtype=np.float32)

	for i, piece in enumerate(pieces):
		if i > 0:
			piece[:, :, :overlap_width] *= slope

		if i < batches - 1:
			piece[:, :, -overlap_width:] *= inv_slope

		result[:, :, segment_width * i:segment_width * i + piece_width] += piece

	return result


def spliceOutputTensor (tensor, keep_margin=False, soft=False, margin_divider=MARGIN_DIVIDER):
	if tensor is None:
		return None
	arr = tensor.cpu().numpy()

	if soft:
		return softSplicePieces(arr, margin_divider)

	return splicePieces(arr, margin_divider, keep_margin=keep_margin)


def randomSliceImage (source, target, width):	# (256, w, 3), (256, w, labels)
	ratio = source.shape[0] // target.shape[0]
	tw = width // ratio

	sliced_source, sliced_target = None, None

	# fill zeros for right residue
	if target.shape[1] < tw:
		sliced_target = np.zeros((target.shape[0], tw, target.shape[2]), dtype=np.float32)
		sliced_target[:, :target.shape[1], :] = target

		sliced_source = np.ones((source.shape[0], width, source.shape[2]), dtype=np.float32)
		sliced_source[:, :source.shape[1], :] = source
	else:
		target_width = target.shape[1]
		target_start = random.randint(0, target_width - tw)
		source_start = target_start * ratio

		sliced_target = target[:, target_start:target_start + tw, :]
		sliced_source = source[:, source_start:source_start + width, :]

	return sliced_source, sliced_target


def iterateSliceImage (source, target, width, overlapping = 0.25, crop_margin = 0):
	ratio = source.shape[0] // target.shape[0]
	tw = width // ratio
	step = math.floor(width - overlapping * source.shape[0])

	for x in range(crop_margin, source.shape[1] - crop_margin, step):
		sliced_source = np.ones((source.shape[0], width, source.shape[2]), dtype=np.float32)
		sliced_target = np.zeros((target.shape[0], tw, target.shape[2]), dtype=np.float32)

		tx = x // ratio
		if x + width <= source.shape[1]:
			sliced_source = source[:, x:x + width, :]
			sliced_target = target[:, tx:tx + tw, :]
		else:
			# fill zeros for right residue
			sliced_source[:, :source.shape[1] - x, :] = source[:, x:, :]
			sliced_target[:, :target.shape[1] - tx, :] = target[:, tx:, :]

		yield sliced_source, sliced_target


def maskToAlpha (mask, frac_y = False):	# mask: [fore(h, w), back(h, w)]
	fore = mask[0] * 255
	fore = np.stack([np.zeros(mask[0].shape, np.float32), fore], axis = 2)
	fore = np.uint8(np.clip(fore, 0, 255))

	return fore


def encodeImageBase64 (image):
	fp = io.BytesIO()
	image.save(fp, PIL.Image.registered_extensions()['.png'])
	image_code = 'data:image/png;base64,' + base64.b64encode(fp.getvalue()).decode('ascii')

	return image_code


def gaugeToRGB (gauge, frac_y=False):	# gauge: [Y(h, w), K(h, w)]
	mapy = gauge[0] * 8 + 128
	mapk = gauge[1] * 127 + 128

	result = None
	if frac_y:
		B, R = np.modf(mapy)
		result = np.stack([B * 256, mapk, R], axis=2)
	else:
		result = np.stack([np.zeros(mapy.shape, np.float32), mapk, mapy], axis=2)
	result = np.uint8(np.clip(result, 0, 255))

	return result
