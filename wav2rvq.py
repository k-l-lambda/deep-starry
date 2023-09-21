
# convert wav files to rvq format

import sys
import os
import logging
import librosa
from tqdm import tqdm
import argparse

import starry.utils.env
from starry.rvq.encodec import sampling_rate
from starry.rvq.rvqFormat import RVQFile



logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='audio directory path')

	args = parser.parse_args()

	for root, dirs, files in os.walk(args.source):
		waves = [f for f in files if f.endswith('.wav')]
		if len(waves) == 0:
			continue

		for file in tqdm(waves, desc=f'Converting {root}'):
			file_path = os.path.join(root, file)
			#logging.info('Converting: %s', file_path)

			audio, sr = librosa.core.load(file_path, sr=sampling_rate, mono=True)
			rvq = RVQFile.fromAudio(audio, sr)

			target_path = file_path.replace('.wav', '.rvq')
			with open(target_path, 'wb') as file:
				rvq.save(file)


if __name__ == "__main__":
	main()
