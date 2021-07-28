
from enum import Enum



class SemanticElementType(Enum):
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

	MAX					= 19,

	def __int__(self):
		return self.value[0]


JOINT_SEMANTIC_ELEMENT_TYPES = list(map(int, [
	SemanticElementType.BOS,
	SemanticElementType.NoteheadS0,
	SemanticElementType.NoteheadS1,
	SemanticElementType.NoteheadS2,
	SemanticElementType.Rest0,
	SemanticElementType.Rest1,
	SemanticElementType.Rest2,
	SemanticElementType.Rest3,
	SemanticElementType.Rest4,
	SemanticElementType.Rest5,
	SemanticElementType.Rest6,
	SemanticElementType.vline_Stem,
]))


JOINT_SOURCE_SEMANTIC_ELEMENT_TYPES = list(map(int, [
	SemanticElementType.NoteheadS0,
	SemanticElementType.NoteheadS1,
	SemanticElementType.NoteheadS2,
	SemanticElementType.Rest0,
	SemanticElementType.Rest1,
	SemanticElementType.Rest2,
	SemanticElementType.Rest3,
	SemanticElementType.Rest4,
	SemanticElementType.Rest5,
	SemanticElementType.Rest6,
	SemanticElementType.vline_Stem,
]))


JOINT_TARGET_SEMANTIC_ELEMENT_TYPES = list(map(int, [
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
]))
