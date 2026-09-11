'''`_shard_interleave` must decorrelate batches without breaking archive-read locality.

Both halves are load-bearing and pull in opposite directions, so each is pinned here: a plain
permutation mixes batches but thrashes the LRU handle cache (measured 3.1x feeder cost on a 256-shard
corpus), while sorted order holds one handle and gives batches whatever was adjacent on disk.
'''

import random

from starry.midi.data.seq2seq2 import _shard_interleave


N_SHARDS = 64
PER_SHARD = 40
WINDOW = 8


def _corpus (n_shards=N_SHARDS, per_shard=PER_SHARD):
	'''Index i belongs to shard i // per_shard, so shard membership is a contiguous block -- the
	adverse case, where sorted order means "one shard at a time".'''
	indices = list(range(n_shards * per_shard))
	return indices, (lambda i: i // per_shard)


def _run (window=WINDOW, seed=0):
	indices, shard_of = _corpus()
	return _shard_interleave(indices, shard_of, random.Random(seed), window), indices, shard_of


def test_is_a_permutation_losing_and_duplicating_nothing ():
	out, indices, _ = _run()
	assert sorted(out) == sorted(indices)
	assert len(out) == len(indices)


def test_is_deterministic_in_the_rng ():
	# Both spawn ranks and every resume must agree on the order.
	assert _run(seed=7)[0] == _run(seed=7)[0]
	assert _run(seed=7)[0] != _run(seed=8)[0]


def test_open_shard_count_never_exceeds_the_window ():
	'''The locality half: walking the order, the set of shards with reads still to come is bounded.'''
	out, _, shard_of = _run()
	remaining = {}
	for i in out:
		remaining[shard_of(i)] = remaining.get(shard_of(i), 0) + 1

	live = set()
	peak = 0
	for i in out:
		shard = shard_of(i)
		live.add(shard)
		remaining[shard] -= 1
		if remaining[shard] == 0:
			live.discard(shard)
		peak = max(peak, len(live))
	assert peak <= WINDOW, f'{peak} shards open at once, window is {WINDOW}'


def test_a_batch_draws_from_many_shards ():
	'''The mixing half: sorted order would put one shard in a batch; this must not.'''
	out, _, shard_of = _run()
	batch = 16
	spans = [len({shard_of(i) for i in out[k:k + batch]})
		for k in range(0, len(out) - batch, batch)]
	mean_span = sum(spans) / len(spans)
	assert mean_span > WINDOW * 0.6, f'batches span only {mean_span:.1f} shards on average'

	_, indices, _ = _run()
	sorted_spans = [len({shard_of(i) for i in indices[k:k + batch]})
		for k in range(0, len(indices) - batch, batch)]
	assert mean_span > 4 * (sum(sorted_spans) / len(sorted_spans))


def test_window_of_one_degenerates_to_shard_at_a_time ():
	out, _, shard_of = _run(window=1)
	# Consecutive runs of one shard: the number of shard CHANGES equals the shard count.
	changes = sum(1 for a, b in zip(out, out[1:]) if shard_of(a) != shard_of(b))
	assert changes == N_SHARDS - 1


def _mean_span (out, shard_of, batch=16):
	# FULL batches only: a short tail slice would understate the span it is averaged into.
	spans = [len({shard_of(i) for i in out[k:k + batch]})
		for k in range(0, len(out) - len(out) % batch - batch + 1, batch)]
	return sum(spans) / len(spans)


def test_mixing_grows_with_the_window ():
	"""The window is the dial, so the ordering it produces must actually respond to it.

	Not pinned to an absolute figure: the achievable span falls as shards drain and `active` shrinks
	toward the end of a pass, so the mean over a whole pass sits well below the batch size even at
	window = shard count (measured 11.2 of a possible 16 here). The monotonicity is the contract.
	"""
	_, _, shard_of = _run()
	spans = [_mean_span(_run(window=w)[0], shard_of) for w in (1, 4, 8, 32, N_SHARDS)]
	assert spans == sorted(spans), spans
	# Window 1 is ~1, not exactly 1: PER_SHARD is not a multiple of the batch, so a batch can straddle
	# one shard boundary. The exact form is pinned by the shard-change count test above.
	assert spans[0] < 1.5
	assert spans[-1] > 10				# window = shard count: most of a batch from distinct shards


def test_order_within_one_shard_is_also_shuffled ():
	"""Mixing across shards is not enough on its own.

	Whatever the sort order MEANS is still present inside a shard -- a shard is a name prefix, so on a
	corpus named by composer or index its members arrive in that order. Draining them in sorted order
	would keep neighbours adjacent in the stream even while other shards interleave between them, so the
	per-shard permutation carries half the decorrelation and is not incidental.
	"""
	out, _, shard_of = _run()
	for shard in (0, 5, N_SHARDS - 1):
		members = [i for i in out if shard_of(i) == shard]
		assert len(members) == PER_SHARD
		# Neither ascending NOR descending: groups are built in ascending order and drained with pop(),
		# so an unshuffled group emerges exactly reversed -- which `!= sorted(members)` would accept.
		# Monotonicity in EITHER direction is the tell that the sort order survived.
		assert members != sorted(members), f'shard {shard} drained in sorted order'
		assert members != sorted(members, reverse=True), f'shard {shard} drained in reverse sorted order'


def test_uneven_shards_are_handled ():
	# A real corpus does not divide evenly; the last shard is short.
	indices = list(range(101))
	out = _shard_interleave(indices, lambda i: i // 10, random.Random(1), 4)
	assert sorted(out) == indices
