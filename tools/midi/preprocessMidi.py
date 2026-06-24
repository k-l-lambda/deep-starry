'''
Preprocess MIDI-text data into a NotaGen/bGPT patch artifact.

Counterpart of tools/lilylet/preprocessLilylet.py for the MIDI-text modality. Walks a
directory of MidiText `.txt` files, tokenizes each into event patches, and writes a
single `.pt` artifact (or a sharded index) the MidiPatchy dataset consumes.

Usage:
  python3 tools/midi/preprocessMidi.py CONFIG SOURCE_DIR OUTPUT.pt
  python3 tools/midi/preprocessMidi.py CONFIG SOURCE_DIR OUTPUT.pt --shard-size 20000 --num-workers 8

The patch_size / patch_length defaults come from the config's data.args; a CONFIG of '-'
skips the config and uses built-in defaults (patch_size 16, patch_length 2048).
'''

import argparse
import logging
import sys

from starry.midi.data.patchifier import pack_midi_notagen, pack_midi_notagen_parallel
from starry.utils.config import Configuration


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main ():
	parser = argparse.ArgumentParser()
	parser.add_argument('config', type=str, help="config of MIDI data to preprocess, or '-' for defaults")
	parser.add_argument('source_dir', type=str, help='input directory containing MidiText .txt files')
	parser.add_argument('output_path', type=str, help='output artifact path (.pt); for sharded output this is the index file')
	parser.add_argument('--shard-size', type=int, default=None, help='items per shard; 0/omitted => single file (falls back to data.args.shard_size)')
	parser.add_argument('--num-workers', type=int, default=0, help='parallel worker processes; >0 enables multiprocessing (sharded mode only). 0 = single process.')
	args = parser.parse_args()

	if args.config == '-':
		data_args = {}
	else:
		config = Configuration.createOrLoad(args.config)
		data_args = config['data.args'] or {}

	patch_size = data_args.get('patch_size', 16)
	patch_length = data_args.get('patch_length', 2048)
	shard_size = args.shard_size if args.shard_size is not None else data_args.get('shard_size', 0)

	logging.info('Preprocessing MIDI-text data: %s (patch_size=%s, patch_length=%s, shard_size=%s, num_workers=%s)',
		args.source_dir, patch_size, patch_length, shard_size, args.num_workers)

	if args.num_workers and args.num_workers > 0 and shard_size and shard_size > 0:
		artifact = pack_midi_notagen_parallel(
			source_dir=args.source_dir,
			output_path=args.output_path,
			patch_size=patch_size,
			patch_length=patch_length,
			shard_size=shard_size,
			num_workers=args.num_workers,
			log=logging.info,
		)
	else:
		artifact = pack_midi_notagen(
			source_dir=args.source_dir,
			output_path=args.output_path,
			patch_size=patch_size,
			patch_length=patch_length,
			shard_size=shard_size,
		)

	logging.info('Wrote %s', args.output_path)
	logging.info('Files: %s', artifact['stats']['files'])
	logging.info('Dropped lines: %s', artifact['stats']['dropped_total'])
	if artifact.get('stats', {}).get('shards'):
		logging.info('Shards: %s', artifact['stats']['shards'])


if __name__ == '__main__':
	main()
