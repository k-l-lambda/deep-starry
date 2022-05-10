
import sys
import io
import json
import PIL.Image
import cv2
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

from starry.utils.config import Configuration
#from starry.vision.scorePageLayout import PageLayout, RESIZE_WIDTH
from starry.vision.layout_predictor import LayoutPredictor



def showLayout (ax, source, layout):
	interval = layout['interval']
	print('interval:', interval)

	h, w = source.shape[:2]
	rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), layout['theta'] * 180 / np.pi, 1)
	image = cv2.warpAffine(source, rot_mat, (w, h), flags=cv2.INTER_CUBIC)
	ax.imshow(image)

	for area in layout['detection']['areas']:
		area_x, area_y = area['x'], area['y']
		ax.add_patch(patches.Rectangle((area_x, area_y), area['width'], area['height'], fill=False, edgecolor='b'))

		if area['staves'] and area['staves']['middleRhos']:
			for rho in area['staves']['middleRhos']:
				phi1, phi2 = area['staves']['phi1'], area['staves']['phi2']
				ax.add_patch(patches.Rectangle((area_x + phi1, area_y + rho - interval * 2), phi2 - phi1, interval * 4, fill=False, edgecolor='r'))


def reinforceScoreJson (scoreJSON, configPath):
	score = json.load(open(scoreJSON, 'r'))
	page = score['pages'][0]
	page_source = page['source']

	layout = {
		'theta': math.atan2(page_source['matrix'][0], page_source['matrix'][2]) - math.pi / 2,
		'interval': page_source['interval'],
		'detection': page['layout'],
	}
	#print('layout:', layout)

	bytes = requests.get(page['source']['url']).content
	source = np.array(PIL.Image.open(io.BytesIO(bytes)))
	print('source:', source.shape)

	_, axes = plt.subplots(1, 2)
	showLayout(axes[0], source, layout)

	config = Configuration.createOrLoad(configPath, volatile=True)
	predictor = LayoutPredictor(config)

	results = predictor.predictReinforce([bytes], [layout])
	layout2 = next(results)
	#print('layout2:', layout2)
	showLayout(axes[1], source, layout2)

	plt.get_current_fig_manager().full_screen_toggle()
	plt.show()


def predictImage (imagePath, configPath):
	config = Configuration.createOrLoad(configPath, volatile=True)
	predictor = LayoutPredictor(config)

	source = cv2.imread(imagePath)
	file = open(imagePath, 'rb')
	bytes = file.read()
	results = predictor.predictDetection([bytes])
	layout = next(results)
	print('layout:', layout)

	_, axes = plt.subplots(1)
	showLayout(axes, source, layout)

	plt.get_current_fig_manager().full_screen_toggle()
	plt.show()


def main ():
	if sys.argv[1].endswith('.json'):
		reinforceScoreJson(sys.argv[1], sys.argv[2])
	elif sys.argv[1].endswith('.png'):
		predictImage(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
	main()
