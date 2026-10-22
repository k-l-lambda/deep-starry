#!/usr/bin/env python3
'''Portable checks for the measure-interleaved Lilylet <-> midiseq2 feeder (InterleavedSeq2).'''

import os
import shutil
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from starry.midi.data.interleavedSeq2 import InterleavedSeq2  # noqa: E402
from mixed_seq2seq2_check import write_fixture, is_lilylet, is_midi, raises  # noqa: E402


def feeder (root, source_format='midiseq2', line_range=(6, 6), **kwargs):
	kwargs = dict(dict(p_head=1, p_tail=0, random_crop=False), **kwargs)
	return InterleavedSeq2(root, '0/1',
		source_dir='midi' if source_format == 'midiseq2' else 'lyl',
		target_dir='lyl' if source_format == 'midiseq2' else 'midi',
		source_format=source_format, target_format='lilylet' if source_format == 'midiseq2' else 'midiseq2',
		measures_path='metadata/measures.json', vocab_path=os.path.join(root, 'vocab.json'),
		line_range=line_range, **kwargs)


def check_layout (dataset, case, src_is, tgt_is):
	'''Unit structure, supervision mask, and per-unit content, for either direction.'''
	tok = dataset.tokenizer
	ids, mask = case['ids'], case['target_mask']
	assert len(ids) == len(mask)
	assert tok.eos_id not in ids
	assert len(case['units']) == len(case['played'])
	prefix = case['units'][0][0]
	assert prefix == (1 if case['head'] else 0) + case['style']
	assert not any(mask[:prefix])
	_, lyl = dataset._mixed_lilylet_document(case['name'])
	midi = dataset._mixed_midi(case['name'])
	end = prefix
	for (start, sep, stop), played, source_measure in zip(
			case['units'], case['played'], case['source_measures']):
		assert start == end, 'units must be contiguous'
		assert ids[sep] == tok.sep_id and ids[stop - 1] == tok.eom_id
		assert src_is(tok, ids[start:sep]) and tgt_is(tok, ids[sep + 1:stop - 1])
		# Source and <sep> are context; every target token and its <eom> is a label.
		assert not any(mask[start:sep + 1]) and all(mask[sep + 1:stop])
		midi_ids = dataset._mixed_encode_midi(dataset._mixed_midi_text(midi, [played]), False)
		lyl_ids = dataset._mixed_encode_lilylet(lyl[source_measure - 1])
		pair = (midi_ids, lyl_ids) if dataset.source_format == 'midiseq2' else (lyl_ids, midi_ids)
		assert ids[start:sep] == pair[0] and ids[sep + 1:stop - 1] == pair[1]
		# No in-unit bar marks: @measure lines are dropped from a MIDI unit.
		assert tok.eom_id not in ids[start:stop - 1]
		end = stop
	assert end == len(ids)


def main ():
	root = tempfile.mkdtemp(prefix='interleaved-seq2-check-')
	try:
		write_fixture(root)
		forward = feeder(root)
		tok = forward.tokenizer
		case = forward.describe(0)
		# Played order: repeats unroll, so Lilylet bars 1, 2 appear twice.
		assert case['played'] == [1, 2, 3, 4, 5, 6]
		assert case['source_measures'] == [1, 2, 1, 2, 3, 4]
		assert case['head'] and case['tail'] and case['ids'][0] == tok.bos_id
		header_lines, _ = forward._mixed_lilylet_document('s1')
		style = forward._mixed_encode_lilylet(''.join(header_lines) + '\n')
		assert case['style'] == len(style) and case['ids'][1:1 + len(style)] == style
		check_layout(forward, case, is_midi, is_lilylet)
		# Measure 1's MIDI unit carries the file header.
		header_ids = forward._mixed_encode_midi(['ticks_per_beat 1 e 0', 'format_type 1'], False)
		first = case['units'][0]
		assert case['ids'][first[0]:first[0] + len(header_ids)] == header_ids

		reverse = feeder(root, source_format='lilylet')
		check_layout(reverse, reverse.describe(0), is_lilylet, is_midi)

		# Interior crops: no <bos>, the MIDI header absent.
		for mode, kwargs in (('middle', dict(p_head=0, p_tail=0)), ('tail', dict(p_head=0, p_tail=1))):
			for i in range(20):
				ds = feeder(root, line_range=(2, 3), seed=i, **kwargs)
				c = ds.describe(0)
				assert c['mode'] == mode
				check_layout(ds, c, is_midi, is_lilylet)
				if mode == 'tail':
					assert c['tail'] and c['z'] == 6
				if c['a'] > 0:
					assert not c['head'] and c['ids'][0] != tok.bos_id

		# max_tokens trims whole units: from the back on a head crop, from the front on a tail one.
		full = len(case['ids'])
		unit_cost = case['units'][-1][2] - case['units'][-1][0]
		trimmed = feeder(root, max_tokens=full - 1).describe(0)
		assert trimmed['trimmed'] == 1 and trimmed['played'] == [1, 2, 3, 4, 5]
		assert trimmed['ids'] == case['ids'][:full - unit_cost]
		check_layout(forward, trimmed, is_midi, is_lilylet)
		tail = feeder(root, line_range=(4, 4), p_head=0, p_tail=1, style_dropout=1)
		whole = tail.describe(0)
		cut = feeder(root, line_range=(4, 4), p_head=0, p_tail=1, style_dropout=1,
			max_tokens=len(whole['ids']) - 1).describe(0)
		assert cut['played'] == whole['played'][1:] and cut['tail']
		# Never below one unit.
		one = feeder(root, max_tokens=1).describe(0)
		assert len(one['units']) == 1 and one['played'] == [1]

		# style_dropout 1: no block, the first unit right after <bos>.
		bare = feeder(root, style_dropout=1).describe(0)
		assert bare['style'] == 0 and bare['units'][0][0] == 1

		batch = forward.collateBatch([forward[0], reverse[0]])
		assert set(batch) == {'input_ids', 'masks', 'target_mask', 'position_ids'}
		for row, ds in enumerate((forward, reverse)):
			c = ds.describe(0)
			n = len(c['ids'])
			assert batch['input_ids'][row, :n].tolist() == c['ids']
			assert batch['target_mask'][row, :n].tolist() == c['target_mask']
			assert not batch['target_mask'][row, n:].any() and not batch['masks'][row, n:].any()
		assert batch['position_ids'][0].tolist() == list(range(batch['input_ids'].shape[1]))

		assert raises(ValueError, lambda: feeder(root, pos_style='sep'))
		assert raises(ValueError, lambda: feeder(root, pack='split'))
		for bad in (dict(start_jitter=1), dict(transposition_sigma=1), dict(mark_mode='tick'), dict(packed=True)):
			assert raises(ValueError, lambda: feeder(root, **bad)), bad
		assert raises(ValueError, lambda: InterleavedSeq2(root, '0/1', source_format='midiseq2',
			target_format='midiseq2', vocab_path=os.path.join(root, 'vocab.json')))
		print('interleaved_seq2_check: OK')
	finally:
		shutil.rmtree(root)


if __name__ == '__main__':
	main()
