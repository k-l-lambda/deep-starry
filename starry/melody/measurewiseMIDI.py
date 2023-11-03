
from enum import IntEnum



BEAT_UNIT = 240

BEAT_MAX = 16


NOTE_MIN = 1
NOTE_MAX = 88


PITCH_OCTAVE_SIZE = 12
N_OCTAVE = 9


class EventType (IntEnum):
	PAD			= 0,
	NOTE_OFF	= 1,
	NOTE_ON		= 2,
	PEDAL		= 3,
