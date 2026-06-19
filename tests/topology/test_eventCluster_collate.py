import torch
import numpy as np
from starry.topology.data.eventCluster import EventCluster
from starry.topology.event_element import TARGET_FIELDS, EventElementType


def make_entry(n_seq=8, n_augment=4, n_feat=16, seed=42):
	"""Create a mock entry dict mimicking the pickled data format."""
	rng = np.random.RandomState(seed)

	# type: (n_seq,) with BOS at 0, EOS at end, CHORD/REST in between
	types = torch.zeros(n_seq, dtype=torch.float32)
	types[0] = EventElementType.BOS
	types[-1] = EventElementType.EOS
	for i in range(1, n_seq - 1):
		types[i] = EventElementType.CHORD if rng.rand() > 0.3 else EventElementType.REST

	staff = torch.zeros(n_seq, dtype=torch.float32)
	staff[1:-1] = torch.tensor(rng.randint(0, 2, size=n_seq - 2), dtype=torch.float32)

	feature = torch.randn(n_augment, n_seq, n_feat)
	x = torch.randn(n_augment, n_seq) * 100
	pivotX = x + torch.randn(n_augment, n_seq) * 5
	y1 = torch.randn(n_augment, n_seq) * 50
	y2 = y1 + torch.rand(n_augment, n_seq) * 20

	tickDiff = torch.randn(n_seq, n_seq)
	maskT = (torch.rand(n_seq, n_seq) > 0.5).float()
	matrixH = torch.randn((n_seq - 1) ** 2)

	time8th = torch.tensor([4.0])
	confidence = torch.ones(n_seq)

	# order for beading path
	order = torch.arange(n_seq, dtype=torch.long)

	# target fields — each is (n_seq,)
	entry = {
		'type': types,
		'staff': staff,
		'feature': feature,
		'x': x,
		'pivotX': pivotX,
		'y1': y1,
		'y2': y2,
		'tickDiff': tickDiff,
		'maskT': maskT,
		'matrixH': matrixH,
		'time8th': time8th,
		'confidence': confidence,
		'order': order,
		'division': torch.zeros(n_seq, dtype=torch.float32),
		'dots': torch.zeros(n_seq, dtype=torch.float32),
	}

	for field in TARGET_FIELDS:
		if field not in entry:
			entry[field] = torch.zeros(n_seq, dtype=torch.float32)

	return entry


def make_dataset(**kwargs):
	"""Create a minimal EventCluster for testing collateBatch."""
	defaults = dict(
		package=None,
		entries=[],
		device='cpu',
		shuffle=False,
		stability_base=10,
		position_drift=0,
		chaos_exp=-1,
		with_beading=False,
	)
	defaults.update(kwargs)
	return EventCluster(**defaults)


# ── 1. Single-entry exact equivalence ───────────────────────────────

def test_single_entry_exact_equivalence():
	"""collateBatch([entry]) must return the same tensors as _processSingleEntry(entry)."""
	ds = make_dataset(batch_slice=2)
	entry = make_entry(n_seq=8, n_augment=4)

	# run _processSingleEntry directly
	torch.manual_seed(99); np.random.seed(99)
	expected = ds._processSingleEntry({k: v.clone() for k, v in entry.items()})

	# run collateBatch with same seeds
	torch.manual_seed(99); np.random.seed(99)
	actual = ds.collateBatch([{k: v.clone() for k, v in entry.items()}])

	assert set(actual.keys()) == set(expected.keys()), f'Key mismatch: {set(actual.keys()) ^ set(expected.keys())}'
	for k in expected:
		assert torch.equal(actual[k], expected[k]), f'Value mismatch for key {k}'

	print('PASS: test_single_entry_exact_equivalence')


# ── 2. Shape tests ──────────────────────────────────────────────────

def test_batch1_shapes():
	"""batch_size=1: shapes should match batch_slice x n_seq."""
	ds = make_dataset(batch_slice=2)
	result = ds.collateBatch([make_entry(n_seq=8, n_augment=4)])

	assert result['type'].shape == (2, 8)
	assert result['feature'].shape == (2, 8, 16)
	assert result['tickDiff'].shape == (2, 8, 8)
	assert result['matrixH'].shape == (2, 49)
	assert result['time8th'].shape == (2,)
	for field in TARGET_FIELDS:
		assert field in result
		assert result[field].shape == (2, 8)

	print('PASS: test_batch1_shapes')


def test_batch2_same_nseq_shapes():
	"""Two entries, same n_seq — batch dim doubles."""
	ds = make_dataset(batch_slice=2)
	result = ds.collateBatch([make_entry(n_seq=8, seed=1), make_entry(n_seq=8, seed=2)])

	assert result['type'].shape == (4, 8)
	assert result['feature'].shape == (4, 8, 16)
	assert result['tickDiff'].shape == (4, 8, 8)
	assert result['matrixH'].shape == (4, 49)
	assert result['time8th'].shape == (4,)
	for field in TARGET_FIELDS:
		assert result[field].shape == (4, 8)

	print('PASS: test_batch2_same_nseq_shapes')


def test_batch2_different_nseq_shapes():
	"""Two entries, different n_seq — padded to max."""
	ds = make_dataset(batch_slice=2)
	result = ds.collateBatch([make_entry(n_seq=6, seed=1), make_entry(n_seq=10, seed=2)])

	ms = 10
	assert result['type'].shape == (4, ms)
	assert result['feature'].shape == (4, ms, 16)
	assert result['tickDiff'].shape == (4, ms, ms)
	assert result['maskT'].shape == (4, ms, ms)
	assert result['matrixH'].shape == (4, (ms - 1) ** 2)
	assert result['time8th'].shape == (4,)
	for field in TARGET_FIELDS:
		assert result[field].shape == (4, ms)

	print('PASS: test_batch2_different_nseq_shapes')


# ── 3. Padding value correctness ────────────────────────────────────

def test_padding_zeros_type_field():
	"""Padded type positions must equal EventElementType.PAD."""
	ds = make_dataset(batch_slice=2)
	result = ds.collateBatch([make_entry(n_seq=6, seed=1), make_entry(n_seq=10, seed=2)])

	# first 2 rows come from n_seq=6, padded at positions 6..9
	assert (result['type'][:2, 6:] == EventElementType.PAD).all(), 'Padded type must be PAD'

	print('PASS: test_padding_zeros_type_field')


def test_padding_zeros_feature():
	"""Feature padded region must be all zeros."""
	ds = make_dataset(batch_slice=2)
	result = ds.collateBatch([make_entry(n_seq=6, seed=1), make_entry(n_seq=10, seed=2)])

	assert (result['feature'][:2, 6:, :] == 0).all(), 'Padded feature must be 0'

	print('PASS: test_padding_zeros_feature')


def test_padding_zeros_targets():
	"""All TARGET_FIELDS padded region must be zero."""
	ds = make_dataset(batch_slice=2)
	result = ds.collateBatch([make_entry(n_seq=6, seed=1), make_entry(n_seq=10, seed=2)])

	for field in TARGET_FIELDS:
		assert (result[field][:2, 6:] == 0).all(), f'{field} padded region should be 0'

	print('PASS: test_padding_zeros_targets')


# ── 4. Value preservation for matrixH ───────────────────────────────

def test_matrixH_content_preservation():
	"""matrixH original block must survive reshape/pad/flatten."""
	ds = make_dataset(batch_slice=1)
	n1, n2 = 5, 8

	e1 = make_entry(n_seq=n1, n_augment=2, seed=1)
	e2 = make_entry(n_seq=n2, n_augment=2, seed=2)

	# process individually to get reference matrixH values
	torch.manual_seed(0); np.random.seed(0)
	r1 = ds._processSingleEntry({k: v.clone() for k, v in e1.items()})
	torch.manual_seed(1); np.random.seed(1)
	r2 = ds._processSingleEntry({k: v.clone() for k, v in e2.items()})

	# batch together
	torch.manual_seed(0); np.random.seed(0)
	r1b = ds._processSingleEntry({k: v.clone() for k, v in e1.items()})
	torch.manual_seed(1); np.random.seed(1)
	r2b = ds._processSingleEntry({k: v.clone() for k, v in e2.items()})
	combined = ds._padAndConcatenate([r1b, r2b])

	ms = n2  # max_seq = 8
	s_max = ms - 1  # 7

	# entry 1: original (4x4) in top-left of (7x7)
	s1 = n1 - 1  # 4
	mat1_orig = r1['matrixH'].reshape(1, s1, s1)
	mat1_combined = combined['matrixH'][:1].reshape(1, s_max, s_max)
	assert torch.equal(mat1_combined[:, :s1, :s1], mat1_orig), 'matrixH entry1: original block corrupted'
	assert (mat1_combined[:, s1:, :] == 0).all(), 'matrixH entry1: bottom pad must be 0'
	assert (mat1_combined[:, :, s1:] == 0).all(), 'matrixH entry1: right pad must be 0'

	# entry 2: fills entire (7x7), no padding
	mat2_orig = r2['matrixH'].reshape(1, s_max, s_max)
	mat2_combined = combined['matrixH'][1:].reshape(1, s_max, s_max)
	assert torch.equal(mat2_combined, mat2_orig), 'matrixH entry2: should be unchanged'

	print('PASS: test_matrixH_content_preservation')


# ── 5. Value preservation for tickDiff/maskT ────────────────────────

def test_tickDiff_maskT_content_preservation():
	"""tickDiff/maskT original top-left block preserved, padded region zero."""
	ds = make_dataset(batch_slice=1)
	n1, n2 = 5, 8

	e1 = make_entry(n_seq=n1, n_augment=2, seed=1)
	e2 = make_entry(n_seq=n2, n_augment=2, seed=2)

	torch.manual_seed(0); np.random.seed(0)
	r1 = ds._processSingleEntry({k: v.clone() for k, v in e1.items()})
	torch.manual_seed(1); np.random.seed(1)
	r2 = ds._processSingleEntry({k: v.clone() for k, v in e2.items()})

	torch.manual_seed(0); np.random.seed(0)
	r1b = ds._processSingleEntry({k: v.clone() for k, v in e1.items()})
	torch.manual_seed(1); np.random.seed(1)
	r2b = ds._processSingleEntry({k: v.clone() for k, v in e2.items()})
	combined = ds._padAndConcatenate([r1b, r2b])

	for key in ('tickDiff', 'maskT'):
		# entry 1: top-left (n1 x n1) preserved
		assert torch.equal(combined[key][:1, :n1, :n1], r1[key]), f'{key} entry1 original block corrupted'
		assert (combined[key][:1, n1:, :] == 0).all(), f'{key} entry1 bottom pad must be 0'
		assert (combined[key][:1, :, n1:] == 0).all(), f'{key} entry1 right pad must be 0'

		# entry 2: no padding needed
		assert torch.equal(combined[key][1:], r2[key]), f'{key} entry2 should be unchanged'

	print('PASS: test_tickDiff_maskT_content_preservation')


# ── 6. Value preservation for concat order ──────────────────────────

def test_concat_preserves_unpadded_slices():
	"""Non-padded slices of each entry must match individually processed results."""
	ds = make_dataset(batch_slice=2)
	n1, n2 = 6, 10

	e1 = make_entry(n_seq=n1, n_augment=4, seed=1)
	e2 = make_entry(n_seq=n2, n_augment=4, seed=2)

	torch.manual_seed(10); np.random.seed(10)
	r1 = ds._processSingleEntry({k: v.clone() for k, v in e1.items()})
	torch.manual_seed(20); np.random.seed(20)
	r2 = ds._processSingleEntry({k: v.clone() for k, v in e2.items()})

	torch.manual_seed(10); np.random.seed(10)
	r1b = ds._processSingleEntry({k: v.clone() for k, v in e1.items()})
	torch.manual_seed(20); np.random.seed(20)
	r2b = ds._processSingleEntry({k: v.clone() for k, v in e2.items()})
	combined = ds._padAndConcatenate([r1b, r2b])

	B1 = r1['type'].shape[0]  # 2

	# 2D fields: check unpadded slice
	for key in ('type', 'staff', 'x', 'pivotX', 'y1', 'y2'):
		assert torch.equal(combined[key][:B1, :n1], r1[key]), f'{key}: entry1 slice mismatch'
		assert torch.equal(combined[key][B1:, :n2], r2[key]), f'{key}: entry2 slice mismatch'

	# 3D feature
	assert torch.equal(combined['feature'][:B1, :n1, :], r1['feature']), 'feature: entry1 slice mismatch'
	assert torch.equal(combined['feature'][B1:, :n2, :], r2['feature']), 'feature: entry2 slice mismatch'

	# 1D time8th
	assert torch.equal(combined['time8th'][:B1], r1['time8th']), 'time8th: entry1 mismatch'
	assert torch.equal(combined['time8th'][B1:], r2['time8th']), 'time8th: entry2 mismatch'

	print('PASS: test_concat_preserves_unpadded_slices')


# ── 7. Beading tests ────────────────────────────────────────────────

def test_beading_same_nseq():
	"""Beading path with batch_size=2, same n_seq."""
	ds = make_dataset(batch_slice=2, with_beading=True)
	result = ds.collateBatch([make_entry(n_seq=8, seed=10), make_entry(n_seq=8, seed=20)])

	assert result['type'].shape == (4, 8)
	assert 'beading_pos' in result
	assert 'successor' in result
	assert 'matrixH' not in result
	assert result['beading_pos'].shape == (4, 8)
	assert result['successor'].shape == (4, 8)

	print('PASS: test_beading_same_nseq')


def test_beading_different_nseq():
	"""Beading path with different n_seq — padding applies to beading fields."""
	ds = make_dataset(batch_slice=2, with_beading=True)
	result = ds.collateBatch([make_entry(n_seq=6, seed=10), make_entry(n_seq=10, seed=20)])

	ms = 10
	assert result['beading_pos'].shape == (4, ms)
	assert result['successor'].shape == (4, ms)
	assert (result['beading_pos'][:2, 6:] == 0).all(), 'beading_pos padded region should be 0'
	assert (result['successor'][:2, 6:] == 0).all(), 'successor padded region should be 0'

	print('PASS: test_beading_different_nseq')


# ── 8. Mixed augment counts (variable per-entry batch size) ────────

def test_mixed_augment_counts():
	"""Entries with different n_augment produce different per-entry batch sizes."""
	ds = make_dataset(batch_slice=3)
	e1 = make_entry(n_seq=8, n_augment=1, seed=1)   # min(1, 3) = 1
	e2 = make_entry(n_seq=8, n_augment=6, seed=2)   # min(6, 3) = 3

	result = ds.collateBatch([e1, e2])

	total_batch = 1 + 3
	assert result['type'].shape == (total_batch, 8), f'Expected ({total_batch}, 8), got {result["type"].shape}'
	assert result['feature'].shape == (total_batch, 8, 16)
	assert result['time8th'].shape == (total_batch,)

	print('PASS: test_mixed_augment_counts')


def test_mixed_augment_and_nseq():
	"""Different n_augment AND n_seq — both padding and variable batch."""
	ds = make_dataset(batch_slice=2)
	e1 = make_entry(n_seq=5, n_augment=1, seed=1)   # produces 1 sample
	e2 = make_entry(n_seq=9, n_augment=4, seed=2)   # produces 2 samples

	result = ds.collateBatch([e1, e2])

	total_batch = 1 + 2
	ms = 9
	assert result['type'].shape == (total_batch, ms)
	assert result['feature'].shape == (total_batch, ms, 16)
	# padded region for entry1 (n_seq=5)
	assert (result['type'][:1, 5:] == EventElementType.PAD).all()
	assert (result['feature'][:1, 5:, :] == 0).all()

	print('PASS: test_mixed_augment_and_nseq')


# ── 9. Minimal sequence length ──────────────────────────────────────

def test_minimal_nseq_2():
	"""n_seq=2 (only BOS+EOS): matrixH is (1,), tickDiff is (2,2)."""
	ds = make_dataset(batch_slice=1)
	e = make_entry(n_seq=2, n_augment=2, seed=1)
	result = ds.collateBatch([e])

	assert result['type'].shape == (1, 2)
	assert result['matrixH'].shape == (1, 1)
	assert result['tickDiff'].shape == (1, 2, 2)

	print('PASS: test_minimal_nseq_2')


def test_minimal_nseq_2_with_padding():
	"""n_seq=2 padded against n_seq=6."""
	ds = make_dataset(batch_slice=1)
	e1 = make_entry(n_seq=2, n_augment=2, seed=1)
	e2 = make_entry(n_seq=6, n_augment=2, seed=2)

	result = ds.collateBatch([e1, e2])

	ms = 6
	assert result['type'].shape == (2, ms)
	assert result['matrixH'].shape == (2, (ms - 1) ** 2)
	assert result['tickDiff'].shape == (2, ms, ms)

	# entry1 padded region
	assert (result['type'][:1, 2:] == EventElementType.PAD).all()

	print('PASS: test_minimal_nseq_2_with_padding')


def test_minimal_nseq_3_beading():
	"""n_seq=3 with beading (BOS + 1 event + EOS)."""
	ds = make_dataset(batch_slice=1, with_beading=True)
	e = make_entry(n_seq=3, n_augment=2, seed=1)
	result = ds.collateBatch([e])

	assert result['type'].shape == (1, 3)
	assert result['beading_pos'].shape == (1, 3)
	assert result['successor'].shape == (1, 3)

	print('PASS: test_minimal_nseq_3_beading')


# ── 10. Source tensor mutation protection ───────────────────────────

def test_source_entry_not_mutated():
	"""collateBatch must not mutate the source entry tensors."""
	ds = make_dataset(batch_slice=2, position_drift=1.0)
	entry = make_entry(n_seq=8, n_augment=4, seed=42)

	# snapshot original values
	orig_feature = entry['feature'].clone()
	orig_x = entry['x'].clone()
	orig_y1 = entry['y1'].clone()

	ds.collateBatch([entry])

	assert torch.equal(entry['feature'], orig_feature), 'feature was mutated'
	assert torch.equal(entry['x'], orig_x), 'x was mutated'
	assert torch.equal(entry['y1'], orig_y1), 'y1 was mutated'

	print('PASS: test_source_entry_not_mutated')


def test_double_collate_deterministic():
	"""Calling collateBatch twice with same seeds must give identical results."""
	ds = make_dataset(batch_slice=2)
	entry = make_entry(n_seq=8, n_augment=4, seed=42)

	torch.manual_seed(77); np.random.seed(77)
	r1 = ds.collateBatch([{k: v.clone() for k, v in entry.items()}])

	torch.manual_seed(77); np.random.seed(77)
	r2 = ds.collateBatch([{k: v.clone() for k, v in entry.items()}])

	for k in r1:
		assert torch.equal(r1[k], r2[k]), f'{k} differs between two calls'

	print('PASS: test_double_collate_deterministic')


# ── 11. time8th value check ─────────────────────────────────────────

def test_time8th_values():
	"""time8th values are correctly repeated and concatenated (no drop)."""
	ds = make_dataset(batch_slice=2, time8th_drop=0)
	e1 = make_entry(n_seq=6, seed=1)
	e2 = make_entry(n_seq=8, seed=2)
	e1['time8th'] = torch.tensor([3.0])
	e2['time8th'] = torch.tensor([7.0])

	result = ds.collateBatch([e1, e2])

	# entry1: batch_slice=2 → [3, 3], entry2: [7, 7]
	assert torch.equal(result['time8th'], torch.tensor([3.0, 3.0, 7.0, 7.0]))

	print('PASS: test_time8th_values')


# ── 12. Empty batch and key mismatch guards ─────────────────────────

def test_empty_batch_raises():
	"""collateBatch([]) must raise with a clear message."""
	ds = make_dataset(batch_slice=2)
	try:
		ds.collateBatch([])
		assert False, 'Should have raised'
	except AssertionError as e:
		assert 'empty' in str(e).lower(), f'Unexpected message: {e}'

	print('PASS: test_empty_batch_raises')


def test_key_mismatch_raises():
	"""_padAndConcatenate must raise when result dicts have different keys."""
	ds = make_dataset(batch_slice=1)

	e1 = make_entry(n_seq=6, n_augment=2, seed=1)
	e2 = make_entry(n_seq=8, n_augment=2, seed=2)

	r1 = ds._processSingleEntry({k: v.clone() for k, v in e1.items()})
	r2 = ds._processSingleEntry({k: v.clone() for k, v in e2.items()})

	# inject an extra key into r2
	r2['bogus'] = torch.zeros(1)

	try:
		ds._padAndConcatenate([r1, r2])
		assert False, 'Should have raised'
	except AssertionError as e:
		assert 'mismatch' in str(e).lower(), f'Unexpected message: {e}'

	print('PASS: test_key_mismatch_raises')


# ── 13. Beading value preservation ──────────────────────────────────

def test_beading_value_preservation():
	"""Beading fields' unpadded slices must match individually processed results."""
	ds = make_dataset(batch_slice=2, with_beading=True)
	n1, n2 = 6, 10

	e1 = make_entry(n_seq=n1, n_augment=4, seed=10)
	e2 = make_entry(n_seq=n2, n_augment=4, seed=20)

	torch.manual_seed(30); np.random.seed(30)
	r1 = ds._processSingleEntry({k: v.clone() for k, v in e1.items()})
	torch.manual_seed(40); np.random.seed(40)
	r2 = ds._processSingleEntry({k: v.clone() for k, v in e2.items()})

	torch.manual_seed(30); np.random.seed(30)
	r1b = ds._processSingleEntry({k: v.clone() for k, v in e1.items()})
	torch.manual_seed(40); np.random.seed(40)
	r2b = ds._processSingleEntry({k: v.clone() for k, v in e2.items()})
	combined = ds._padAndConcatenate([r1b, r2b])

	B1 = r1['type'].shape[0]

	for key in ('beading_pos', 'successor'):
		assert torch.equal(combined[key][:B1, :n1], r1[key]), f'{key}: entry1 unpadded slice mismatch'
		assert torch.equal(combined[key][B1:, :n2], r2[key]), f'{key}: entry2 slice mismatch'
		# padded region
		assert (combined[key][:B1, n1:] == 0).all(), f'{key}: entry1 padded region should be 0'

	print('PASS: test_beading_value_preservation')


# ── main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
	tests = [
		test_single_entry_exact_equivalence,
		test_batch1_shapes,
		test_batch2_same_nseq_shapes,
		test_batch2_different_nseq_shapes,
		test_padding_zeros_type_field,
		test_padding_zeros_feature,
		test_padding_zeros_targets,
		test_matrixH_content_preservation,
		test_tickDiff_maskT_content_preservation,
		test_concat_preserves_unpadded_slices,
		test_beading_same_nseq,
		test_beading_different_nseq,
		test_mixed_augment_counts,
		test_mixed_augment_and_nseq,
		test_minimal_nseq_2,
		test_minimal_nseq_2_with_padding,
		test_minimal_nseq_3_beading,
		test_source_entry_not_mutated,
		test_double_collate_deterministic,
		test_time8th_values,
		test_empty_batch_raises,
		test_key_mismatch_raises,
		test_beading_value_preservation,
	]

	passed = 0
	failed = []
	for test in tests:
		try:
			test()
			passed += 1
		except Exception as e:
			failed.append((test.__name__, e))
			print(f'FAIL: {test.__name__}: {e}')

	print(f'\n{passed}/{len(tests)} passed')
	if failed:
		print('Failed:')
		for name, e in failed:
			print(f'  {name}: {e}')
		exit(1)
	else:
		print('All tests passed!')
