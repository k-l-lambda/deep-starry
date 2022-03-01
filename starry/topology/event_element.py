
from enum import IntEnum
from pickle import NONE



class EventElementType (IntEnum):
	PAD					= 0,
	BOS					= 1,
	EOS					= 2,

	CHORD				= 3,
	REST				= 4,

	MAX					= 5,


class BeamType (IntEnum):
	NONE				= 0,
	Open				= 1,
	Continue			= 2,
	Close				= 3,


class StemDirection (IntEnum):
	NONE				= 0,
	u					= 1,
	d					= 2,


TARGET_FIELDS = [
	'tick', 'division', 'dots', 'beam', 'stemDirection', 'grace', 'timeWarped', 'fullMeasure', 'confidence',
]
