"""Visual test for causal_mask on real EventCluster data.

Loads dataset from bdtopo-writer-20231004 config, prints each sample's
type, staff, beading_pos, fixed, x, and causal mask matrix.
Horizontal axis = sequence position, vertical axis = fields + mask rows.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from starry.utils.config import Configuration
from starry.utils.dataset_factory import loadDataset
from starry.topology.models.beadPicker import _build_causal_mask
from starry.topology.event_element import EventElementType

TYPE_NAMES = {0: 'PAD', 1: 'BOS', 2: 'EOS', 3: 'CHD', 4: 'RST'}


def fmt_val(v, width=5):
	s = str(v)
	return s.rjust(width)


def print_sample(batch, b, seq_len):
	stype = batch['type'][b]
	staff = batch['staff'][b]
	bp = batch['beading_pos'][b]
	x = batch['pivotX'][b] if 'pivotX' in batch else batch['x'][b]

	# Trim trailing PAD for display
	non_pad = (stype != EventElementType.PAD).nonzero(as_tuple=False).squeeze(-1)
	if non_pad.numel() == 0:
		print("  (empty sample)")
		return
	end = non_pad[-1].item() + 1

	is_bos = stype == EventElementType.BOS
	is_fixed = (bp < 0) & ~is_bos
	w = 6

	# Header: position indices
	header = "idx".rjust(12) + ''.join(fmt_val(i, w) for i in range(end))
	print(header)
	print('-' * len(header))

	# Type
	row = "type".rjust(12)
	for i in range(end):
		row += fmt_val(TYPE_NAMES.get(stype[i].item(), '?'), w)
	print(row)

	# Staff
	row = "staff".rjust(12)
	for i in range(end):
		row += fmt_val(staff[i].item(), w)
	print(row)

	# beading_pos
	row = "beading_pos".rjust(12)
	for i in range(end):
		row += fmt_val(bp[i].item(), w)
	print(row)

	# fixed
	row = "fixed".rjust(12)
	for i in range(end):
		row += fmt_val('*' if is_fixed[i].item() else '.', w)
	print(row)

	# x (rounded)
	row = "x".rjust(12)
	for i in range(end):
		row += fmt_val(f'{x[i].item():.1f}', w)
	print(row)

	# Build causal mask for this single sample
	mask = _build_causal_mask(
		stype.unsqueeze(0),
		bp.unsqueeze(0),
		x.unsqueeze(0),
		strict_causal=True,
	)[0]  # (seq, seq)

	print()
	print("  Causal mask (rows=query, cols=key, 1=attend, .=block):")

	# Column header
	col_hdr = "q\\k".rjust(12) + ''.join(fmt_val(i, 3) for i in range(end))
	print(col_hdr)

	for qi in range(end):
		label = f"[{qi}]"
		if is_fixed[qi]:
			label += f"bp={bp[qi].item()}"
		row = label.rjust(12)
		for ki in range(end):
			ch = '1' if mask[qi, ki].item() else '.'
			row += fmt_val(ch, 3)
		print(row)

	# Verify: non-fixed non-PAD non-BOS rows should have full attention to all non-PAD
	for qi in range(end):
		if stype[qi] == EventElementType.PAD:
			continue
		if is_fixed[qi] or is_bos[qi]:
			continue
		for ki in range(end):
			if stype[ki] == EventElementType.PAD:
				continue
			assert mask[qi, ki].item(), \
				f"Non-fixed query {qi} should attend to non-PAD key {ki}"

	# Verify: BOS is visible to all non-PAD queries (key), but only sees itself (query)
	for qi in range(end):
		if stype[qi] == EventElementType.PAD:
			continue
		for ki in range(end):
			if not is_bos[ki]:
				continue
			assert mask[qi, ki].item(), \
				f"Query {qi} should always attend to BOS key {ki}"
	for qi in range(end):
		if not is_bos[qi]:
			continue
		for ki in range(end):
			if ki == qi:
				assert mask[qi, ki].item(), "BOS should attend to itself"
			else:
				assert not mask[qi, ki].item(), \
					f"BOS query {qi} should not attend to key {ki}"

	# Verify: fixed elements — cross-segment isolated, within-segment causal
	non_pad_fixed = is_fixed[:end] & (stype[:end] != EventElementType.PAD)
	fixed_indices = non_pad_fixed.nonzero(as_tuple=False).squeeze(-1)
	if fixed_indices.numel() >= 2:
		bp_vals = bp[fixed_indices]
		sorted_order = bp_vals.argsort()
		sorted_fixed = fixed_indices[sorted_order]
		sorted_x = x[sorted_fixed]

		# Build segments
		segments = [[0]]
		for k in range(1, len(sorted_fixed)):
			if sorted_x[k] > sorted_x[k - 1]:
				segments[-1].append(k)
			else:
				segments.append([k])

		print(f"\n  Fixed elements: {fixed_indices.tolist()}")
		print(f"  Sorted by bp: {sorted_fixed.tolist()}")
		print(f"  Segments (by x monotonicity): {segments}")

		# Verify: fixed queries cannot attend to non-fixed keys (strict_causal),
		# except BOS and EOS which are global context keys always visible.
		is_eos_row = stype[:end] == EventElementType.EOS
		non_fixed_non_global = (~is_fixed[:end]) & (stype[:end] != EventElementType.PAD) & (~is_bos[:end]) & (~is_eos_row)
		for pi in range(len(sorted_fixed)):
			q_idx = sorted_fixed[pi].item()
			for ki in non_fixed_non_global.nonzero(as_tuple=False).squeeze(-1).tolist():
				assert not mask[q_idx, ki].item(), \
					f"Strict causal: fixed query {q_idx} should not attend to non-fixed key {ki}"
			for ki in is_eos_row.nonzero(as_tuple=False).squeeze(-1).tolist():
				assert mask[q_idx, ki].item(), \
					f"Strict causal: fixed query {q_idx} should always see EOS key {ki}"

		# Verify: cross-segment fully isolated, within-segment causal
		for si, seg_i in enumerate(segments):
			for sj, seg_j in enumerate(segments):
				for pi in seg_i:
					q_idx = sorted_fixed[pi].item()
					for pj in seg_j:
						k_idx = sorted_fixed[pj].item()
						if q_idx == k_idx:
							assert mask[q_idx, k_idx].item(), \
								f"Fixed element {q_idx} should see itself"
						elif si != sj:
							assert not mask[q_idx, k_idx].item(), \
								f"Cross-segment: query {q_idx}(seg{si}) should not see key {k_idx}(seg{sj})"
						elif pi < pj:
							assert not mask[q_idx, k_idx].item(), \
								f"Within-segment causal: query {q_idx} should not see later key {k_idx}"

	print("  [OK] Mask verified.\n")


def main():
	DATA_DIR = os.environ.get('DATA_DIR')
	config = Configuration.create('configs/bdtopo-writer-20231004.local.yaml')
	train, = loadDataset(config, data_dir=DATA_DIR, device='cpu', splits='0/1')

	n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100

	it = iter(train)
	count = 0
	for batch_idx, batch in enumerate(it):
		batch_size = batch['type'].shape[0]
		seq_len = batch['type'].shape[1]

		for b in range(batch_size):
			print(f"{'='*60}")
			print(f"Sample {count} (batch {batch_idx}, item {b}), seq_len={seq_len}")
			print(f"{'='*60}")
			print_sample(batch, b, seq_len)
			count += 1
			if count >= n_samples:
				return


if __name__ == '__main__':
	main()
