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


def assert_midiseq2 (lines, path, tokenizer, threshold=0.2):
	"""Fail loudly when `lines` are not midiseq2. -> the unknown-token fraction.

	Both translate tools take `--input <file>.midiseq2.txt` and hand it straight to `encode_lines`,
	which resolves every token with `lookup.get(token, unknown)`. That default is right for a stray
	token inside real midiseq2 and WRONG for a whole file in the wrong language: MidiText -- what
	`tools/midiToTextSegments.ts` writes, and what piano0909/segs holds -- shares the event keywords
	(`note_on`, `set_tempo`) but writes the fields as raw hex words rather than midiseq2's `#26 $34`
	pitch/velocity and `E040` elapse tokens. MEASURED on a real segment against the 582-token
	midiseq2 vocabulary: 5222 of 9835 tokens (53.1%) resolve to <unknown>, and NOTHING raises. The
	run completes, burns its GPU time and publishes a file built from a source the model could not
	read.

	So the check is the unknown RATE, not a filename or a header sniff. A `.txt` suffix says nothing
	(both languages use it), and a header sniff would pass a file whose first two lines happen to be
	`ticks_per_beat`/`format_type` -- which MidiText's are. The rate separates the two languages by
	two orders of magnitude, so any threshold in between works; 0.2 is set well above real
	midiseq2's own rate and far below MidiText's.

	The fix for a MidiText input is to CONVERT it first (midiToSeq2Server.ts with kind `text`, which
	is what translate_piano0909.sh --source segs does), not to relax this.
	"""
	lookup, unknown = tokenizer.id_by_token, tokenizer.unknown_id
	total = miss = 0
	examples = []
	for line in lines:
		if line.startswith('@'):		# directives are control, never looked up
			continue
		for token in line.split():
			total += 1
			if lookup.get(token, unknown) == unknown:
				miss += 1
				if len(examples) < 8:
					examples.append(token)
	if not total:
		raise ValueError(f'{path}: no tokens to translate')
	rate = miss / total
	if rate >= threshold:
		raise ValueError(
			f'{path}: {miss}/{total} tokens ({rate:.1%}) are outside the midiseq2 vocabulary, '
			f'e.g. {examples}. This looks like MidiText rather than midiseq2 -- convert it first '
			f'(midiToSeq2Server.ts, kind `text`; translate_piano0909.sh --source segs does this), '
			f'because encode_lines would silently map every one of them to <unknown>.')
	return rate
