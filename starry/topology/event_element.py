
from enum import IntEnum
from pickle import NONE

from ..modules.classInt import pclassDims



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


NONE_LIST = lambda l: [None if x == 'NONE' else x for x in l]

BeamType_values = NONE_LIST([x.name for x in BeamType])
StemDirection_values = NONE_LIST([x.name for x in StemDirection])


STAFF_MAX = 4


# d_feature = d_division + d_dots + d_beam + d_stemDirection + d_grace + d_tremoloCatcher
FEATURE_DIM = 7 + 2 + 3 + 2 + 1 + 1

TARGET_DIMS_LEGACY = dict(
	tick=1,
	division=7,
	dots=3,
	beam=4,
	stemDirection=3,
	grace=1,
	timeWarped=1,
	fullMeasure=1,
	fake=1,
)
TARGET_DIM_LEGACY = sum(TARGET_DIMS_LEGACY.values())

TARGET_DIMS = dict(
	tick=1,
	division=9,
	dots=3,
	beam=4,
	stemDirection=3,
	grace=1,
	timeWarped=1,
	fullMeasure=1,
	fake=1,
)
TARGET_FIELDS = list(TARGET_DIMS.keys())
TARGET_DIM = sum(TARGET_DIMS.values())


TIME8TH_MAX = 16


VTICK_BASE = 1920 * 4
VTICK_DIMS = pclassDims(VTICK_BASE)

TARGET_DIMS_V3 = dict(
	vtick=VTICK_DIMS,
	division=9,
	dots=3,
	beam=4,
	stemDirection=3,
	grace=1,
	timeWarped=1,
	fullMeasure=1,
	fake=1,
)
TARGET_FIELDS_V3 = list(TARGET_DIMS_V3.keys())
TARGET_DIM_V3 = sum(TARGET_DIMS_V3.values())
