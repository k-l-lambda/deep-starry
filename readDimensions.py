
import os
import re
import magic
import argparse
from fs import open_fs



def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('dir', type=str)
	parser.add_argument('-r', '--round', type=int, default=8)
	parser.add_argument('-l', '--low_bound', type=int, default=64, help='downsample resolution low bound')
	parser.add_argument('-o', '--output', type=str, default=None)

	args = parser.parse_args()

	items = []

	fs = open_fs(args.dir)
	files = fs.listdir('/')
	for filename in files:
		with fs.open(filename, 'rb') as file:
			info = magic.from_buffer(file.read(0x10000))
			#print('info:', filename, info)
			capture = re.match('.*,\s(\d+)x(\d+),.*', info) or re.match('.*,\s(\d+)\sx\s(\d+),.*', info)
			if capture is not None:
				w, h = int(capture[1]), int(capture[2])
				#print('dimension:', w, h, filename)

				rw, rh = w - w % args.round, h - h % args.round
				down = 0
				while min(rw, rh) > args.low_bound:
					items.append((filename, f'{rw}x{rh}', down, h, w))

					down += 1
					rw //= 2
					rh //= 2
					rw, rh = rw - rw % args.round, rh - rh % args.round
			else:
				print('no dimensions:', filename)

	items.sort(key=lambda item: item[1])
	#print('items:', items)

	output_path = args.output or os.path.join(args.dir, 'dimensions.csv')
	with open(output_path, 'w') as output:
		output.write('name,size,down,height,width\n')
		for item in items:
			output.write(','.join(map(str, item)))
			output.write('\n')

	print(f'Done, {len(items)} items')


if __name__ == '__main__':
	main()
