
import os
import sys
import argparse
import logging
import librosa
import torch
from tqdm import tqdm

import starry.utils.env
from starry.rmvpe.inference import RMVPE



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


RMVPE_MODEL = os.environ.get('RMVPE_MODEL')

SR = 16000
HOP_SIZE = 512


def extractAudio (pe, audio: str):
	sample, _ = librosa.load(audio, sr=SR)
	f0, uv = pe.get_pitch(sample, sample.shape[0] // HOP_SIZE, dict(audio_sample_rate=SR, hop_size=HOP_SIZE), interp_uv=True)

	target_path = audio.replace('.wav', '.pitch.pt')
	torch.save(dict(f0=f0, uv=uv), target_path)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('source', type=str, help='audio directory path')

	args = parser.parse_args()

	pe = RMVPE(RMVPE_MODEL)

	for root, dirs, files in os.walk(args.source):
		audios = [name for name in files if name.endswith('.wav')]
		for audio in tqdm(audios, desc=f'Extracting {root}'):
			extractAudio(pe, os.path.join(root, audio))


	logging.info('Done.')



if __name__ == '__main__':
	main()
