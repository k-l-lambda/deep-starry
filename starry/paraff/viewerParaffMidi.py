
import logging
import matplotlib.pyplot as plt
import json
import math

from ..melody.midiEncoder import encodeMeasurewise



TYPE2SHAPE = {
	1: 's',
	2: '<',
	3: '^',
}

MEASURE2COLOR = {
	-1: 'y',
	0: 'r',
	1: 'c',
}


class ParaffMidiViewer:
	def __init__ (self, config):
		self.vocab = config['_vocab']


	def show (self, data):
		for ib, batch in enumerate(data):
			logging.info('batch: %d', ib)

			self.showExample(batch)


	def showExample (self, batch, inspection=None):
		plt.figure(0)
		plt.get_current_fig_manager().full_screen_toggle()

		time, type_, pitch, measure = batch['time'][0], batch['type'][0], batch['pitch'][0], batch['measure'][0]
		is_entity = type_ != 0

		time, type_, pitch, measure = time[is_entity], type_[is_entity], pitch[is_entity], measure[is_entity]

		output_id, body_mask = batch['output_id'], batch['body_mask']
		output_id, body_mask = output_id[0], body_mask[0]
		output_id_body = output_id[body_mask]
		tokens = [self.vocab[id] for id in output_id]
		tokens_body = [self.vocab[id] for id in output_id_body]

		#print('paraff full:', ' '.join((tokens)))
		print('paraff body:', ' '.join((tokens_body)))

		for t, y, p, m in zip(time, type_, pitch, measure):
			plt.plot(t, p, TYPE2SHAPE.get(y.item(), 'o'), color=MEASURE2COLOR.get(m.item(), 'k'), markerfacecolor='none', markersize=16)

		if inspection is not None:
			pred_midi = inspection['pred_midi'][0][is_entity]

			for t, y, p, m, pm in zip(time, type_, pitch, measure, pred_midi):
				plt.plot(t, p, TYPE2SHAPE.get(y.item(), 'o'), color=MEASURE2COLOR.get(m.item(), 'k'), alpha=pm.item(), markersize=16)

		midi = encodeMeasurewise(batch, 0)
		with open('./test/viewerParaffMidi-midi.json', 'w') as f:
			json.dump(midi, f)

		if inspection is not None:
			plt.figure(1)
			self.showParaff(batch, inspection)

		plt.show()


	def showParaff (self, batch, inspection):
		is_entity = batch['output_id'][0] != 0
		target_id = batch['output_id'][0][is_entity]
		pred_id = inspection['pred_id'][0][is_entity]
		truth = inspection['truth'][0][is_entity]

		def format_coord (x, y):
			x, y = math.floor(x), math.floor(y)
			token = self.vocab[y] if y < len(self.vocab) else f'[{y}]'
			pred = pred_id[x][y]
			return f'(seq_i, token, pred) = {x},\t{token:>8},\t{pred:8.4f}'
		plt.gca().format_coord = format_coord

		plt.pcolormesh(pred_id.transpose(0, 1).numpy(), cmap='RdBu', vmin=-25, vmax=30)
		plt.xlabel('seq')
		plt.ylabel('vocab id')
		plt.xticks([i for i, _ in enumerate(target_id)], [self.vocab[id] + ('' if truth[i] else ' *') for i, id in enumerate(target_id)], rotation=-60)
		plt.yticks([i for i, _ in enumerate(self.vocab)], [token for token in self.vocab])
		plt.colorbar()
