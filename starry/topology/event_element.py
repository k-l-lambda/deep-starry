
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


STAFF_MAX = 4


# d_feature = d_division + d_dots + d_beam + d_stemDirection + d_grace
FEATURE_DIM = 7 + 2 + 3 + 2 + 1

TARGET_DIMS = (
	1,	# tick
	7,	# division
	3,	# dots
	4,	# beam
	3,	# stemDirection
	1,	# grace
	1,	# timeWarped
	1,	# fullMeasure
	1,	# confidence
)
TARGET_DIM = sum(TARGET_DIMS)
