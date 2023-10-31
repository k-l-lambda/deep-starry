
import logging
import matplotlib.pyplot as plt
import json

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


	def showExample (self, batch, pred=None):
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
			plt.plot(t, p, TYPE2SHAPE.get(y.item(), 'o'), color=MEASURE2COLOR.get(m.item(), 'k'))
		#plt.plot(time, pitch, 'o')
		#plt.plot(time, measure, '-')

		midi = encodeMeasurewise(batch, 0)
		with open('./test/viewerParaffMidi-midi.json', 'w') as f:
			json.dump(midi, f)

		plt.show()
