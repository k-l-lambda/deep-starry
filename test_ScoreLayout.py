
import sys
import io
import json
import PIL.Image
import requests
import matplotlib.pyplot as plt
import numpy as np

from starry.vision.scorePageLayout import PageLayout, RESIZE_WIDTH



def main ():
	score = json.load(open(sys.argv[1], 'r'))
	page = score['pages'][0]
	#print('layout:', page['layout'])

	bytes = requests.get(page['source']['url']).content
	source = np.array(PIL.Image.open(io.BytesIO(bytes)))
	print('source:', source.shape)

	plt.imshow(source)
	plt.get_current_fig_manager().full_screen_toggle()
	plt.show()


if __name__ == '__main__':
	main()
