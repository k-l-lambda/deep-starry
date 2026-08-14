#!/usr/bin/env python3
'''Portable checks for repeat-aware mixed Lilylet/midiseq2 Seq2Seq2 feeding.'''

import copy
import json
import os
import shutil
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from starry.midi.data.seq2seq2 import Seq2Seq2  # noqa: E402
from starry.midi.data.unifiedSeq2Tokenizer import write_unified_vocab  # noqa: E402


LYL = '''%Classical
%Test, Anon
%Keyboard
[staves ","]
[instrument-1 "Piano" "Pno."]

\\key c \\major \\time 4/4 \\clef "treble" c4 d e f |
g4 a b c' |
c'4 b a g |
f4 e d c |
'''


def midi_text (count=6):
	lines = ['ticks_per_beat 1 e 0', 'format_type 1']
	for measure in range(1, count + 1):
		lines.extend((f'@measure {measure}', '@tick 0', 'note_on C1 #48 $50', 'note_off C1 #48'))
	return '\n'.join(lines) + '\n'


def record (mapping):
	return {'ok': True, 'repeat_aligned': True, 'has_repeats': True, 'measures': [
		{'index': i, 'start_tick': (i - 1) * 1920, 'source_measure': source}
		for i, source in enumerate(mapping, 1)
	]}


def write_fixture (root, metadata=None):
	os.makedirs(os.path.join(root, 'lyl'))
	os.makedirs(os.path.join(root, 'midi'))
	os.makedirs(os.path.join(root, 'metadata'))
	with open(os.path.join(root, 'lyl', 's1.lyl'), 'w') as f:
		f.write(LYL)
	with open(os.path.join(root, 'midi', 's1.midiseq2.txt'), 'w') as f:
		f.write(midi_text())
	with open(os.path.join(root, 'metadata', 'measures.json'), 'w') as f:
		json.dump({'s1': metadata or record([1, 2, 1, 2, 3, 4])}, f)
	write_unified_vocab(os.path.join(root, 'vocab.json'))


def feeder (root, source_format='midiseq2', line_range=(6, 6)):
	return Seq2Seq2(root, '0/1',
		source_dir='midi' if source_format == 'midiseq2' else 'lyl',
		target_dir='lyl' if source_format == 'midiseq2' else 'midi',
		source_format=source_format, target_format='lilylet' if source_format == 'midiseq2' else 'midiseq2',
		measures_path='metadata/measures.json', vocab_path=os.path.join(root, 'vocab.json'),
		line_range=line_range, p_head=1, p_tail=0, random_crop=False, pos_style='sep')


def raises (error, function):
	try:
		function()
	except error:
		return True
	return False


def is_lilylet (tok, ids):
	'''Every id is either a shared control or Lilylet content — never midiseq2 content.'''
	lo, size = tok.lilylet_offset, tok.blocks['lilylet']['size']
	return bool(ids) and all(i < 16 or lo <= i < lo + size for i in ids)


def is_midi (tok, ids):
	lo, size = tok.midiseq2_offset, tok.blocks['midiseq2']['size']
	return bool(ids) and all(i < 16 or lo <= i < lo + size for i in ids)


def rewrite_metadata (root, metadata):
	with open(os.path.join(root, 'metadata', 'measures.json'), 'w') as f:
		json.dump({'s1': metadata}, f)


def main ():
	root = tempfile.mkdtemp(prefix='mixed-seq2seq2-check-')
	try:
		write_fixture(root)
		forward = feeder(root)
		case = forward.describe(0)
		tok = forward.tokenizer
		assert len(forward) == 1
		assert forward._measure_maps['s1'] == [1, 2, 1, 2, 3, 4]
		assert case['source_measures'] == [1, 2, 3, 4, 5, 6]
		assert case['target_measures'] == [1, 2, 1, 2, 3, 4]
		assert case['target_range'] == (0, 4)
		assert case['ids'][case['sep']] == tok.sep_id
		# Controls are SHARED (ids < 16) in both arms; only the content ranges are modality-specific,
		# so "every id on the MIDI side is above the MIDI offset" is deliberately no longer true.
		assert is_midi(tok, case['ids'][1:case['sep']])
		assert is_lilylet(tok, case['ids'][case['sep'] + 2:-1])
		assert case['ids'][0] == tok.bos_id
		assert case['ids'][-1] == tok.eos_id
		# source_eom defaults off and the Lilylet target has no <eom>, so this direction emits none.
		assert tok.eom_id not in case['ids']
		assert tok.eom_id in forward._mixed_encode_midi(forward._mixed_midi_text(
			forward._mixed_midi('s1'), [1, 2]), True)
		header_ids = forward._mixed_encode_midi(['ticks_per_beat 1 e 0', 'format_type 1'], False)
		assert case['ids'][1:1 + len(header_ids)] == header_ids
		midi = forward._mixed_midi('s1')
		assert forward._mixed_midi_text(midi, [1])[:2] == ['ticks_per_beat 1 e 0', 'format_type 1']
		assert all(not line.startswith(('ticks_per_beat', 'format_type'))
			for line in forward._mixed_midi_text(midi, [2]))
		assert case['ids'][case['sep'] + 1] == tok.bos_id
		assert len(case['positions']) == len(case['ids'])

		reverse = feeder(root, source_format='lilylet', line_range=(2, 2))
		case = reverse.describe(0)
		assert case['source_measures'] == [1, 2]
		assert case['target_measures'] == [1, 2, 3, 4]
		assert is_lilylet(tok, case['ids'][1:case['sep']])
		assert is_midi(tok, case['ids'][case['sep'] + 2:-1])
		# Same shared wrappers in the reverse direction — the target EOS is not modality-specific.
		assert (case['ids'][0], case['ids'][case['sep'] + 1]) == (tok.bos_id, tok.bos_id)
		assert case['ids'][-1] == tok.eos_id
		assert tok.eom_id in case['ids'][case['sep'] + 1:]

		# Lilylet ids are byte VALUES before remapping, so a newline or an ASCII digit must not leak
		# through as a raw local id.
		lyl_body = ''.join(reverse._mixed_lilylet_measures('s1')[:2])
		local_ids = reverse.lilylet_tokenizer.encode(lyl_body)
		assert reverse._mixed_encode_lilylet(lyl_body) == [tok.lilylet_id(i) for i in local_ids]
		assert 10 in local_ids and tok.lilylet_id(10) == 18
		assert is_lilylet(tok, reverse._mixed_encode_lilylet(lyl_body))

		# A Lilylet bar absent from the played mapping is a rejected crop, not a fatal parse error.
		rewrite_metadata(root, record([2, 3, 4, 2, 3, 4]))
		sparse = feeder(root, source_format='lilylet', line_range=(1, 1))
		draws = iter(((0, 1), (1, 2)))
		sparse._mixed_pick = lambda count, rng: next(draws)
		case = sparse.describe(0)
		assert case['source_measures'] == [2]
		assert case['target_measures'] == [1, 4]
		rewrite_metadata(root, record([1, 2, 1, 2, 3, 4]))

		batch = forward.collateBatch([forward[0], forward[0]])
		assert set(batch) == {'input_ids', 'masks', 'target_mask', 'sep_index', 'position_ids'}
		for row, sep in enumerate(batch['sep_index'].tolist()):
			assert not batch['target_mask'][row, :sep + 1].any()
			assert int(batch['target_mask'][row].sum()) == int(batch['masks'][row].sum()) - sep - 1

		bad = record([1, 2, 1, 2, 3, 4])
		bad['repeat_aligned'] = False
		bad['measures'][2]['source_measure'] = None
		rewrite_metadata(root, bad)
		assert raises(RuntimeError, lambda: feeder(root))

		bad = copy.deepcopy(bad)
		bad['repeat_aligned'] = True
		rewrite_metadata(root, bad)
		assert raises(ValueError, lambda: feeder(root))

		bad = record([1, 2, 1, 2, 3, 4])
		bad['measures'][0]['index'] = 2
		rewrite_metadata(root, bad)
		assert raises(ValueError, lambda: feeder(root))

		bad = record([1, 2, 1, 2, 3, 4])
		bad['ok'] = False
		rewrite_metadata(root, bad)
		assert raises(ValueError, lambda: feeder(root))
	finally:
		shutil.rmtree(root)

	print('mixed Seq2Seq2 checks passed')


if __name__ == '__main__':
	main()
