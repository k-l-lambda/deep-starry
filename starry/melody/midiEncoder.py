
import torch
import torch.nn.functional as F

from .measurewiseMIDI import EventType



PITCH2PEDAL = {
	89: 64,
	90: 67,
}


def encodeMeasurewise (batch, index):
	type_, pitch, strength, time = batch['type'], batch['pitch'], batch['strength'], batch['time']
	n_seq = (type_[index] != 0).sum().item()

	type_, pitch, strength, time = type_[index][:n_seq], pitch[index][:n_seq], strength[index][:n_seq], time[index][:n_seq]
	deltatime = F.pad(time[1:] - time[:-1], (1, 0), value=time[0]).int()

	#print('type_:', type_, n_seq)
	#print('time:', time)
	#print('deltaTime:', deltaTime)

	events = []
	events.append(dict(deltaTime=0, type='meta', subtype='trackName', text='notes'))
	for i in range(n_seq):
		y, p, s, t = type_[i].item(), pitch[i].item(), strength[i].item(), deltatime[i].item()

		if y == EventType.NOTE_ON:
			events.append(dict(deltaTime=t, channel=0, type='channel', subtype='noteOn', noteNumber=p + 20, velocity=int(s * 127)))
		elif y == EventType.NOTE_OFF:
			events.append(dict(deltaTime=t, channel=0, type='channel', subtype='noteOff', noteNumber=p + 20, velocity=0))
		elif y == EventType.PEDAL:
			events.append(dict(deltaTime=t, channel=0, type='channel', subtype='controller', controllerType=PITCH2PEDAL[p], value=int(s * 127)))
		else:
			raise Exception(f'unexpected event type: {y}, at[{i}]')

	events.append(dict(deltaTime=100, type='meta', subtype='endOfTrack'))

	midi = dict(
		header=dict(
			formatType=1,
			ticksPerBeat=480,
			trackCount=2,
		),
		tracks=[
			[
				dict(deltaTime=0, type='meta', subtype='trackName', text='meta'),
				dict(deltaTime=0, type='meta', subtype='setTempo', microsecondsPerBeat=480e+3),
				dict(deltaTime=0, type='meta', subtype='endOfTrack'),
			],
			events,
		],
	)

	return midi
