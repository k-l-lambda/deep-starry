
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging

from .score_semantic import ScoreSemantic
from .chromaticChannels import composeChromaticMap



def scoreAnno (heatmap, srcimg, labels, target=[]):
	#print(f'heatmap:{heatmap.shape} srcimg{srcimg.shape}')

	labels_count = len(labels)
	#labels_count = min(labels_count, 4)

	jout = ScoreSemantic(heatmap, labels)
	#print(jout.dump_json())
	marks = jout.marks
	# print(f"jout.marks: {len(marks)}")
	clean_bgr = np.zeros_like(heatmap[0])

	if len(target) > 0:
		_, ax = plt.subplots(labels_count+2, 3)
	else:
		_, ax = plt.subplots(labels_count+2, 2)

	for i in range(labels_count):
		ax[i][1].imshow(clean_bgr)

		if marks[i].get('points') is not None:
			points = marks[i]['points']
			if len(points) > 0:
				ax[i][1].plot(points[:, 0], points[:, 1], 'g.')
		elif marks[i].get('vlines') is not None:
			lines = marks[i]['vlines']
			if len(lines) > 0:
				ax[i][1].vlines(lines[:, 0], lines[:, 1], lines[:, 2], color = 'g')
		elif marks[i].get('rectangles') is not None:
			rectangles = marks[i]['rectangles']
			if len(rectangles) > 0:
				for rect in rectangles:
					rc = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='b', facecolor='g')
					ax[i][1].add_patch(rc)
		elif marks[i].get('boxes') is not None:
			boxes = marks[i]['boxes']
			if len(boxes) > 0:
				for box in boxes:
					#print('box:', box)
					angle = box[2] * math.pi / 180
					wcos = box[1][0] * math.cos(angle)
					wsin = box[1][0] * math.sin(angle)
					hcos = box[1][1] * math.cos(angle)
					hsin = box[1][1] * math.sin(angle)
					rx = wcos - hsin
					ry = wsin + hcos
					rc = patches.Rectangle((box[0][0] - rx / 2, box[0][1] - ry / 2), box[1][0], box[1][1], angle=box[2], linewidth=1, edgecolor='b', facecolor='g')
					ax[i][1].add_patch(rc)

		ax[i][1].set_ylabel(labels[i], fontsize='small', rotation='horizontal')
		ax[i][0].imshow(heatmap[i],cmap='jet')

		hm_max = np.max(heatmap[i])
		ax[i][0].set_ylabel(f'({hm_max})', fontsize='small', rotation='horizontal')

		if len(target) > 0:
			ax[i][2].imshow(target[...,i], cmap='jet')

	ax[labels_count][1].imshow(clean_bgr)

	all_points = [m.get('points') for m in marks if len(m.get('points', [])) > 0]
	if len(all_points) > 0:
		all_points = np.concatenate(all_points)
		print(f'all_points-{all_points.shape} heatmap-{heatmap.shape}')
		ax[labels_count][1].plot(all_points[:, 0], all_points[:,1], 'g.')

	ax[labels_count][1].set_ylabel('max', fontsize='small', rotation='horizontal')
	ax[labels_count][0].imshow(np.max(heatmap, axis=0), cmap='jet')
	if len(target) > 0:
		ax[labels_count][2].imshow(np.max(target, axis=-1), cmap='jet')
		ax[labels_count+1][2].imshow(srcimg, cmap='gray')

	ax[labels_count+1][1].set_ylabel('srcimg', fontsize='small', rotation='horizontal')
	ax[labels_count+1][1].imshow(srcimg, cmap='gray')
	#ax[labels_count+1][0].imshow(srcimg, cmap='gray')
	ax[labels_count+1][0].imshow(srcimg)
	plt.get_current_fig_manager().full_screen_toggle()
	plt.show()
	plt.close()


def scoreAnnoChromatic (feature, targets, predictions=None):
	#print('scoreAnnoChromatic:', feature.shape, targets.shape, predictions.shape)
	with_pred = predictions is not None
	_, ax = plt.subplots(2, 2)

	colored_target = composeChromaticMap(targets)
	cv2.imwrite('./temp/feature.png', feature)
	cv2.imwrite('./temp/target.png', colored_target)

	#print('colored_target:', colored_target.shape, targets.dtype)
	ax[0][0].imshow(colored_target, cmap='gray')
	if with_pred:
		colored_pred = composeChromaticMap(predictions)
		ax[0][1].imshow(colored_pred)
		cv2.imwrite('./temp/pred.png', colored_pred)

	ax[1][0].imshow(feature)
	ax[1][1].set_ylabel('feature', rotation='vertical')
	ax[1][1].imshow(feature, cmap='gray')

	plt.get_current_fig_manager().full_screen_toggle()
	plt.show()
	plt.close()


def gaugeToRGB (gauge, frac_y = False):	# gauge: [Y(h, w), K(h, w)]
	mapy = gauge[0] * 8 + 128
	mapk = gauge[1] * 127 + 128

	result = None
	if frac_y:
		B, R = np.modf(mapy)
		result = np.stack([B * 256, mapk, R], axis = 2)
	else:
		result = np.stack([np.zeros(mapy.shape, np.float32), mapk, mapy], axis = 2)
	result = np.uint8(np.clip(result, 0, 255))

	return result


image_staff_frame = None

def gaugeToFrame (gauge):	# gauge: [Y(h, w), K(h, w)]
	global image_staff_frame
	if image_staff_frame is None:
		image_staff_frame = cv2.imread('./staff-frame.png')
		image_staff_frame = image_staff_frame[:, :, ::-1]

	# integrate to make X map
	map_x = np.zeros(gauge.shape[1:], np.float32)
	half_height = gauge.shape[1] // 2
	for x in range(gauge.shape[2]):
		map_x[half_height, x] = x + 100

	for y in range(half_height):	# X[y - 1] = X[y] - K[y]
		map_x[half_height - y - 1, :] = map_x[half_height - y, :] - gauge[1, half_height - y, :]

	for y in range(1, half_height):	# X[y] = X[y - 1] + K[y]
		map_x[half_height + y, :] = map_x[half_height + y - 1, :] + gauge[1, half_height + y, :]

	map_y = gauge[0] * 8 + image_staff_frame.shape[0] // 2

	return cv2.remap(image_staff_frame, map_x, map_y, cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)


def gaugeToFrameFeature (gauge, feature):
	frame = gaugeToFrame(gauge)

	inv_feature = np.uint8((255 - np.minimum(np.float32(feature) + 64, 255)))

	frame[:, :, 0] = np.maximum(inv_feature, frame[:, :, 0])
	frame[:, :, 1] = np.maximum(inv_feature, frame[:, :, 1])
	frame[:, :, 2] = np.maximum(inv_feature, frame[:, :, 2])

	return frame


def scoreAnnoGauge (feature, targets, predictions=None):
	with_pred = predictions is not None
	fig, ax = plt.subplots(4, 2)

	cv2.imwrite('./temp/feature.png', feature)
	cv2.imwrite('./temp/target.png', gaugeToRGB(targets))

	if len(targets) >= 3:
		ax[0][0].imshow(targets[2])
	else:
		ax[0][0].imshow(feature)
	ax[0][1].set_ylabel('feature', rotation='vertical')
	ax[0][1].imshow(feature, cmap='gray')

	ax[1][1].set_ylabel('Y')
	ax[2][1].set_ylabel('K')

	y = ax[1][0].imshow(targets[0])
	k = ax[2][0].imshow(targets[1])
	fig.colorbar(y, ax = ax[1][0], shrink=0.4, pad=0.01)
	fig.colorbar(k, ax = ax[2][0], shrink=0.4, pad=0.01)

	frame = gaugeToFrameFeature(targets, feature)
	ax[3][0].imshow(frame)

	if with_pred:
		y = ax[1][1].imshow(predictions[0])
		k = ax[2][1].imshow(predictions[1])
		fig.colorbar(y, ax = ax[1][1], shrink=0.4, pad=0.01)
		fig.colorbar(k, ax = ax[2][1], shrink=0.4, pad=0.01)
		cv2.imwrite('./temp/pred.png', gaugeToRGB(predictions))

		pred_frame = gaugeToFrameFeature(predictions, feature)
		ax[3][1].imshow(pred_frame)

	plt.get_current_fig_manager().full_screen_toggle()
	plt.show()
	plt.close()


class DatasetViewer:
	def __init__(self, config, chromatic=True, gauge_mode=False):
		self.chromatic = chromatic
		self.gauge_mode = gauge_mode

		self.labels = config['data.args.labels']


	def show(self, data_set):
		for batch, (feature, target) in enumerate(data_set):
			bsize = len(feature)
			for i in range(bsize):
				index = batch * bsize + i
				logging.info(index)

				img = feature[i].cpu().numpy()
				hm = target[i].cpu().numpy()
				img = np.moveaxis(img, 0, -1)
				img = img.reshape(img.shape[:2]) if img.shape[2] == 1 else img.reshape(img.shape[:3])
				img = np.clip(img, 0., 1.)
				img = np.uint8(img * 255)
				#print('img shape2:', img.shape)

				if self.gauge_mode:
					scoreAnnoGauge(img, hm)
				else:
					if self.chromatic:
						scoreAnnoChromatic(img, hm)
					else:
						hm = np.uint8(hm * 255)
						scoreAnno(hm, img, labels = self.labels)
