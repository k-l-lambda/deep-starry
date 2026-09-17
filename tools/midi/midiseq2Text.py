'''Text-level midiseq2 rules, importable without torch.

`translateMidiseq2.py` applies these while writing a pair and `repairStrayTerminators.py` applies
them to already-published files. They live here so the repair tool -- which only rewrites lines --
does not have to import the model stack to agree with the generator.
'''


def reconcile_terminators (out_lines, src_lines):
	"""Stop the output arm claiming an ending the source arm does not have. -> (lines, changed).

	`close_final_measure` and `annotate_source` each decide their own last line, and that
	independence is deliberate -- a trimmed run whose source was nonetheless consumed to the last
	line legitimately ends on `@measure` here and on `end_of_track` there. MEASURED over 5801
	published piano0909 pairs: 224 in that direction, and it is correct.

	The INVERSE is not. `end_of_track` on the output arm asserts the piece is over, so a pair whose
	source arm ends on `@measure N` says the source still had music the output claims to have
	finished. MEASURED at 9 of 5801, and every one traced to the same shape: the model emitted a
	terminator, `annotate_source` then dropped a source tail (`33406005b8`: furthest 1553/1611, 17
	notes past the closing bar line) or the align stop had already cut the run (`43975e4b3a`:
	stopped after 2795/35231 source lines). Either way the terminator was never earned.

	So it is removed and the bar closed the way every other trimmed run closes it: a bare
	`@measure N` that opens nothing. The bar COUNT does not move -- `end_of_track` was closing the
	last bar and the directive now closes it instead, and neither is counted -- so the annotation
	already computed against that count stays valid and needs no recomputation.

	Runs on the composed TEXT of both arms, after both are known, because that is the first point
	where either arm can see the other's decision.
	"""
	if not out_lines or not src_lines:
		return out_lines, False
	if out_lines[-1] != 'end_of_track' or src_lines[-1] == 'end_of_track':
		return out_lines, False
	kept = out_lines[:-1]
	# The count of `@measure` directives IS the last bar's number, so the next one closes it.
	nth = sum(1 for l in kept if l.startswith('@measure')) + 1
	return kept + [f'@measure {nth}'], True
