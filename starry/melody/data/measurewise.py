
import numpy as np

from ..measurewiseMIDI import BEAT_UNIT



def normalFactor (sigma = 1):
	return np.exp(np.random.randn() * sigma)


class MeasurewiseScore:
	def __init__ (self, data):
		self.measures = data['measures']
		self.events = data['events']


	def __len__ (self):
		return len(self.measures)


	@staticmethod
	def parseEventData (data):
		t, f1, f2 = data
		t = t >> 4

		pitch = [64, 67].index(f1) + 89 if t == 0xb else max(1, f1 - 20)

		return dict(
			type=[8, 9, 0xb].index(t) + 1,	# EventType
			pitch=pitch,
			strength=f2 / 127,
		)


	@property
	def endTime (self):
		return self.events[-1]['time']


	def slice (self, start, length, pre=0, n_seq=256, aug_time_index=None):
		pre = min(start, pre)
		m0 = start - pre

		measure_ticks = [measure['tick'] for measure in self.measures[m0:start + length]]
		t0 = measure_ticks[pre]
		measure_ticks = [t - t0 for t in measure_ticks]

		measure_durations = [measure['duration'] for measure in self.measures[m0:start + length]]

		events = [event for event in self.events if event['measure'] >= m0 and event['measure'] < start + length]
		if len(events) == 0:
			return events

		n_event_current = sum(1 for e in events if e['measure'] == start)
		if n_event_current == 0:
			return []

		n_event_pre = sum(1 for e in events if e['measure'] < start)
		n_event_pre_expected = min(max(n_seq - int(n_event_current * 1.5), 0), int((n_seq // 8) * normalFactor()))
		if n_event_pre_expected < n_event_pre:
			events = events[n_event_pre - n_event_pre_expected:]
		#print(dict(n_event_current=n_event_current, n_event_pre=n_event_pre, n_event_pre_expected=n_event_pre_expected, n_event=len(events)))

		aug_time_i = aug_time_index % len(events[0].get('aug_times')) if (aug_time_index is not None and events[0].get('aug_times') is not None) else None

		return [dict(
			MeasurewiseScore.parseEventData(event['data']),
			measure=event['measure'] - start,
			tick=event['tick'] + measure_ticks[event['measure'] - m0],
			beat=event['tick'] // BEAT_UNIT,
			phase=float(event['tick'] % measure_durations[event['measure'] - m0]) / measure_durations[event['measure'] - m0],
			time=event['aug_times'][aug_time_i] if aug_time_i is not None else event['time'],
		) for event in events]

