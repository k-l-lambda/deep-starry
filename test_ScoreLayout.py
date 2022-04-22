
import sys
import io
import json
import PIL.Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import math

from starry.vision.scorePageLayout import PageLayout, RESIZE_WIDTH



def showLayout (ax, source, layout):
	ax.imshow(source)


def main ():
	score = json.load(open(sys.argv[1], 'r'))
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

	plt.get_current_fig_manager().full_screen_toggle()
	plt.show()


if __name__ == '__main__':
	main()
