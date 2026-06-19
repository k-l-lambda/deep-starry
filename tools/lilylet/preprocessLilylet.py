import argparse
import logging
import os
import sys

from starry.lilylet.data.patchifier import pack_lilylet_notagen, pack_lilylet_notagen_parallel
from starry.utils.config import Configuration


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str, help='config of Lilylet data to preprocess')
	parser.add_argument('source_dir', type=str, help='input directory containing .lyl files')
	parser.add_argument('output_path', type=str, help='output artifact path (.pt); for sharded output this is the index file')
	parser.add_argument('--shard-size', type=int, default=None, help='items per shard; 0/omitted => single file (falls back to data.args.shard_size)')
	parser.add_argument('--num-workers', type=int, default=0, help='parallel worker processes; >0 enables multiprocessing (sharded mode only). 0 = single process.')
	args = parser.parse_args()

	config = Configuration.createOrLoad(args.config)
	tokenizer_path = config['data.args']['tokenizer_path']
	patch_size = config['data.args'].get('patch_size', 16)
	patch_length = config['data.args'].get('patch_length', 2048)
	patch_stream = config['data.args'].get('patch_stream', True)
	shard_size = args.shard_size if args.shard_size is not None else config['data.args'].get('shard_size', 0)

	logging.info('Preprocessing Lilylet data: %s (shard_size=%s, num_workers=%s)', args.source_dir, shard_size, args.num_workers)
	if args.num_workers and args.num_workers > 0 and shard_size and shard_size > 0:
		artifact = pack_lilylet_notagen_parallel(
			source_dir=args.source_dir,
			output_path=args.output_path,
			tokenizer_path=tokenizer_path,
			patch_size=patch_size,
			patch_length=patch_length,
			patch_stream=patch_stream,
			shard_size=shard_size,
			num_workers=args.num_workers,
			log=logging.info,
		)
	else:
		artifact = pack_lilylet_notagen(
			source_dir=args.source_dir,
			output_path=args.output_path,
			tokenizer_path=tokenizer_path,
			patch_size=patch_size,
			patch_length=patch_length,
			patch_stream=patch_stream,
			shard_size=shard_size,
		)
	logging.info('Wrote %s', args.output_path)
	logging.info('Files: %s', artifact['stats']['files'])
	logging.info('Unknown total: %s', artifact['stats']['unknown_total'])
	if artifact.get('stats', {}).get('shards'):
		logging.info('Shards: %s', artifact['stats']['shards'])


if __name__ == '__main__':
	main()
