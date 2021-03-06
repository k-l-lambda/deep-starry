
from enum import IntEnum



class SemanticElementType(IntEnum):
	BOS					= 0,
	PAD					= 1,

	NoteheadS0			= 2,
	NoteheadS1			= 3,
	NoteheadS2			= 4,
	GraceNoteheadS0		= 5,
	vline_Stem			= 6,
	Flag3				= 7,
	BeamLeft			= 8,
	BeamContinue		= 9,
	BeamRight			= 10,
	Dot					= 11,
	Rest0				= 12,
	Rest1				= 13,
	Rest2				= 14,
	Rest3				= 15,
	Rest4				= 16,
	Rest5				= 17,
	Rest6				= 18,

	TimeD2				= 19,
	TimeD4				= 20,
	TimeD8				= 21,
	TimeN1				= 22,
	TimeN2				= 23,
	TimeN3				= 24,
	TimeN4				= 25,
	TimeN5				= 26,
	TimeN6				= 27,
	TimeN7				= 28,
	TimeN8				= 29,
	TimeN9				= 30,
	TimeN10				= 31,
	TimeN11				= 32,
	TimeN12				= 33,

	MAX					= 34,


JOINT_SEMANTIC_ELEMENT_TYPES = [
	SemanticElementType.BOS,
	SemanticElementType.NoteheadS0,
	SemanticElementType.NoteheadS1,
	SemanticElementType.NoteheadS2,
	SemanticElementType.GraceNoteheadS0,
	SemanticElementType.Rest0,
	SemanticElementType.Rest1,
	SemanticElementType.Rest2,
	SemanticElementType.Rest3,
	SemanticElementType.Rest4,
	SemanticElementType.Rest5,
	SemanticElementType.Rest6,
	SemanticElementType.vline_Stem,
]


JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES = [
	SemanticElementType.NoteheadS0,
	SemanticElementType.NoteheadS1,
	SemanticElementType.NoteheadS2,
	SemanticElementType.GraceNoteheadS0,
	SemanticElementType.Rest0,
	SemanticElementType.Rest1,
	SemanticElementType.Rest2,
	SemanticElementType.Rest3,
	SemanticElementType.Rest4,
	SemanticElementType.Rest5,
	SemanticElementType.Rest6,
	SemanticElementType.vline_Stem,
]


JOINT_TARGET_SEMANTIC_ELEMENT_TYPES = [
	SemanticElementType.BOS,
	SemanticElementType.NoteheadS0,
	SemanticElementType.Rest0,
	SemanticElementType.Rest1,
	SemanticElementType.Rest2,
	SemanticElementType.Rest3,
	SemanticElementType.Rest4,
	SemanticElementType.Rest5,
	SemanticElementType.Rest6,
	SemanticElementType.vline_Stem,
]


ROOT_SEMANTIC_ELEMENT_TYPES = [
	SemanticElementType.NoteheadS0,
	SemanticElementType.Rest0,
	SemanticElementType.Rest1,
	SemanticElementType.Rest2,
	SemanticElementType.Rest3,
	SemanticElementType.Rest4,
	SemanticElementType.Rest5,
	SemanticElementType.Rest6,
	SemanticElementType.vline_Stem,
]


ROOT_NOTE_SEMANTIC_ELEMENT_TYPES = [
	SemanticElementType.NoteheadS0,
	SemanticElementType.NoteheadS1,
	SemanticElementType.NoteheadS2,
	SemanticElementType.GraceNoteheadS0,
	SemanticElementType.Rest0,
	SemanticElementType.Rest1,
	SemanticElementType.Rest2,
	SemanticElementType.Rest3,
	SemanticElementType.Rest4,
	SemanticElementType.Rest5,
	SemanticElementType.Rest6,
	SemanticElementType.vline_Stem,
]


STAFF_MAX = 8
