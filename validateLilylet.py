
import sys
import os
import json
import argparse
import logging
import torch

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset


# workaround cuda unavailable issue
torch.cuda.is_available()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


DATA_DIR = os.environ.get('DATA_DIR')


def loadTokenView (tokenizer_path):
	if tokenizer_path is None or not os.path.exists(tokenizer_path):
		return None
	with open(tokenizer_path, 'r', encoding='utf-8') as f:
		artifact = json.load(f)
	return {entry['id']: entry['token'] for entry in artifact['vocab']}


def displayToken (token_id, id_to_token):
	if id_to_token is None:
		return str(token_id)
	token = id_to_token.get(token_id)
	if token is None:
		return f'<{token_id}>'
	if token == '\n':
		return r'\n'
	if token == '\t':
		return r'\t'
	if token == ' ':
		return '·'
	return token


def showBatch (batch, index, id_to_token, max_patches):
	patches = batch['input_patches'][0]
	masks = batch['input_masks'][0]
	# Supervision mask (optional): 1 where the patch is a prediction target, 0 over the
	# prompt + <bos> boundary + padding. Absent for legacy datasets.
	targets = batch['input_targets'][0] if 'input_targets' in batch else None
	n_patches, patch_size = patches.shape

	if targets is not None:
		logging.info('batch %d: patches=%s masks=%s real=%d supervised=%d', index,
			tuple(patches.shape), tuple(masks.shape), int(masks.sum()), int(targets.sum()))
	else:
		logging.info('batch %d: patches=%s masks=%s real=%d', index,
			tuple(patches.shape), tuple(masks.shape), int(masks.sum()))

	limit = n_patches if max_patches <= 0 else min(max_patches, n_patches)
	for p in range(limit):
		ids = [int(x) for x in patches[p].tolist()]
		rendered = ' '.join(displayToken(x, id_to_token) for x in ids)
		tgt = f' tgt={int(targets[p])}' if targets is not None else ''
		print(f'  [{p:04d}] mask={int(masks[p])}{tgt} | {rendered}')
	if limit < n_patches:
		print(f'  ... ({n_patches - limit} more patches)')


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str)
	parser.add_argument('-s', '--splits', type=str, default='0/1')
	parser.add_argument('-dv', '--device', type=str, default='cpu')
	parser.add_argument('-n', '--batches', type=int, default=2, help='number of batches to show, 0 for all')
	parser.add_argument('-p', '--patches', type=int, default=40, help='max patches per batch to print, 0 for all')

	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config, volatile=True)
	config['data.splits'] = args.splits
	config['data.batch_size'] = 1

	id_to_token = loadTokenView(config['data.args.tokenizer_path'])

	data, = loadDataset(config, data_dir=DATA_DIR, device=args.device)

	for i, batch in enumerate(data):
		if args.batches > 0 and i >= args.batches:
			break
		showBatch(batch, i, id_to_token, args.patches)


if __name__ == '__main__':
	main()
