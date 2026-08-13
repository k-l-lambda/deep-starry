'''Does a packed corpus feed the model exactly what the loose one does?

The whole point of `packMidiseq2Shards.py` is that it changes storage and nothing else. That claim is
only worth having if it is checked at the level the model sees, so the decisive test here is
EQUIVALENCE: for every index, a zip-backed feeder and a directory-backed feeder must agree on ids, sep,
positions AND skip. Anything weaker (file counts, sizes, a spot-check of one sample) would let a routing
bug — an off-by-one shard, a name colliding across arms, a stale handle — reach training as an
unattributable accuracy difference.

Run against a matched pair of roots:

    python tests/midi/seq2seq2_archive_check.py --dir-root /tmp/t202608-hex \\
        --zip-root /tmp/t202608-hex-packed

Sections
    1. manifest agrees with the archives on disk
    2. every archive holds both arms for every name it claims (no half pairs)
    3. the two feeders enumerate the same names in the same order
    4. EQUIVALENCE: ids / sep / positions / skip identical per index
    5. collateBatch output identical tensor-for-tensor
    6. the LRU stays bounded and still serves correct content after eviction
    7. auto-detection picks the right source for each root
'''

import argparse
import json
import os
import sys
import zipfile

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from starry.midi.data import seq2seq2 as S			# noqa: E402
from starry.midi.data.seq2seq2 import Seq2Seq2, _DirSource, _ZipSource, _make_source	# noqa: E402

failures = []


def check (label, ok, detail=''):
	print(('  ok   ' if ok else '  FAIL ') + label + (f'   {detail}' if detail else ''))
	if not ok:
		failures.append(label)


def feeder (root, source_dir, target_dir, split='0/1', **args):
	'''Deterministic by default: equivalence is only meaningful when the crop does not re-randomize.'''
	options = dict(source_dir=source_dir, target_dir=target_dir, mark_mode='tick',
		line_range=[20, 256], random_crop=False)
	options.update(args)
	(dataset,) = Seq2Seq2.load(root, options, splits=split)
	return dataset


def check_manifest (zip_root):
	'''1. The manifest must describe the archives that exist, not the ones the packer intended.'''
	print('\n== 1. manifest')
	path = os.path.join(zip_root, _ZipSource.MANIFEST)
	check('index.json present', os.path.isfile(path), path)
	if not os.path.isfile(path):
		return None
	with open(path, 'r', encoding='utf-8') as f:
		manifest = json.load(f)
	on_disk = {n[:-4] for n in os.listdir(zip_root) if n.endswith('.zip')}
	claimed = set(manifest['shards'])
	check('every claimed shard has an archive', claimed <= on_disk,
		f'{len(claimed - on_disk)} missing')
	check('every archive is claimed', on_disk <= claimed, f'{len(on_disk - claimed)} unclaimed')
	total = sum(len(v) for v in manifest['shards'].values())
	check('sample count matches the shard lists', manifest['samples'] == total,
		f'{manifest["samples"]} vs {total}')
	# the shard key must actually be the name's prefix, or read routing goes to the wrong archive
	bad = [(s, n) for s, names in manifest['shards'].items() for n in names
		if n[:manifest['shard_chars']] != s]
	check('every name sits in the shard its prefix names', not bad, f'{len(bad)} misfiled')
	print(f'  {len(manifest["shards"])} shards, {total} samples, arms {manifest["arms"]}')
	return manifest


def check_pairs (zip_root, manifest, arms):
	'''2. A shard that holds one arm of a sample but not the other would silently shrink the corpus.'''
	print('\n== 2. archive contents')
	if manifest is None:
		return
	missing = 0
	extra = 0
	for shard, names in sorted(manifest['shards'].items()):
		with zipfile.ZipFile(os.path.join(zip_root, f'{shard}.zip')) as z:
			entries = set(z.namelist())
		want = {f'{arm}/{name}' for arm in arms for name in names}
		missing += len(want - entries)
		extra += len(entries - want)
	check('both arms present for every claimed name', missing == 0, f'{missing} missing entries')
	check('no unclaimed entries in any archive', extra == 0, f'{extra} extra entries')


def check_names (dir_ds, zip_ds, subset):
	'''3. The split filter is POSITIONAL, so a different order silently repartitions train/val.

	When the packed root is a SUBSET of the loose one (`--subset`, the normal case when only a few
	shards are packed), the name lists cannot match — so what is checked instead is that the zip's
	names are a subset in the same relative order, and equivalence is aligned by NAME rather than by
	index. Comparing by index across differently-sized lists would compare different samples and pass
	or fail for the wrong reason.
	'''
	print('\n== 3. enumeration')
	if subset:
		zset = set(zip_ds.names)
		check('zip names are a subset of dir names', zset <= set(dir_ds.names),
			f'{len(zset - set(dir_ds.names))} not in dir')
		check('sorted order agrees', zip_ds.names == sorted(zset))
		print(f'  {len(zip_ds.names)} packed of {len(dir_ds.names)} loose')
		return
	check('same name count', len(dir_ds.names) == len(zip_ds.names),
		f'{len(dir_ds.names)} dir vs {len(zip_ds.names)} zip')
	check('same names in the same order', dir_ds.names == zip_ds.names)
	check('same split indices', dir_ds.indices == zip_ds.indices,
		f'{len(dir_ds.indices)} vs {len(zip_ds.indices)}')


def check_equivalence (dir_ds, zip_ds, limit=0):
	'''4. The one that matters: identical ids, sep, positions and skip for the same sample.

	Compared at the SAME INDEX, which requires the same name list. `describe(i)` seeds its rng on the
	index (`seed ^ (index * 2654435761)`), so one sample sitting at index 7 in one dataset and index 300
	in the other draws a DIFFERENT crop — legitimately. Aligning by name alone would therefore report
	differences that are not storage bugs at all.

	So when the pack is a subset, the loose dataset is narrowed to the packed name list first. That is
	not a fudge: the feeder's own logic (`names` -> positional split -> per-index rng) is rerun over one
	agreed list, which is exactly the condition under which the two layouts must agree.
	'''
	print('\n== 4. equivalence (dir vs zip)')
	if dir_ds.names != zip_ds.names:
		phases, cycle = S.parseFilterStr('0/1')
		dir_ds.names = list(zip_ds.names)
		dir_ds.indices = [i for i in range(len(dir_ds.names)) if i % cycle in phases]
		print(f'  narrowed the loose dataset to the packed {len(dir_ds.names)} names')
	n = min(len(dir_ds.names), len(zip_ds.names))
	if limit:
		n = min(n, limit)
	bad_ids = bad_sep = bad_pos = bad_skip = bad_lines = 0
	for i in range(n):
		a = dir_ds.describe(i)
		b = zip_ds.describe(i)
		# same sample on both sides, or the rest of this comparison means nothing
		if a['name'] != b['name']:
			raise AssertionError(f'alignment bug: {a["name"]} vs {b["name"]}')
		# the raw parsed text first: a wrong archive would differ here before it differed in ids
		if a['source'].lines != b['source'].lines or a['target'].lines != b['target'].lines:
			bad_lines += 1
		if a['ids'] != b['ids']:
			bad_ids += 1
		if a['sep'] != b['sep']:
			bad_sep += 1
		if a['positions'] != b['positions']:
			bad_pos += 1
		if a['skip'] != b['skip']:
			bad_skip += 1
	check('parsed lines identical', bad_lines == 0, f'{bad_lines}/{n} differ')
	check('ids identical', bad_ids == 0, f'{bad_ids}/{n} differ')
	check('sep identical', bad_sep == 0, f'{bad_sep}/{n} differ')
	check('positions identical', bad_pos == 0, f'{bad_pos}/{n} differ')
	check('skip identical', bad_skip == 0, f'{bad_skip}/{n} differ')
	print(f'  compared {n} samples')


def check_collate (dir_ds, zip_ds):
	'''5. Through the collate path the trainer actually uses, tensor for tensor.'''
	print('\n== 5. collateBatch')
	k = min(8, len(dir_ds.indices), len(zip_ds.indices))
	a = dir_ds.collateBatch([dir_ds[i] for i in range(k)])
	b = zip_ds.collateBatch([zip_ds[i] for i in range(k)])
	check('same keys', set(a) == set(b))
	diff = [key for key in a if not torch.equal(a[key], b[key])]
	check('every tensor identical', not diff, str(diff))
	print(f'  batch {tuple(a["input_ids"].shape)}, supervised {int(a["target_mask"].sum())}')


def check_cache_bound (zip_root, source_dir, target_dir):
	'''6. The LRU must actually evict, and must still return correct content once it has.

	A cache that evicts but then serves a stale or wrong entry is worse than one that grows, so this
	re-reads after forcing eviction and compares against a fresh parse.
	'''
	print('\n== 6. LRU bound')
	saved_limit = S._CACHE_LIMIT
	saved_cache = dict(S._FILE_CACHE)
	try:
		S._FILE_CACHE.clear()
		S._CACHE_LIMIT = 4			# below the 2 files/sample the feeder touches over several samples
		ds = feeder(zip_root, source_dir, target_dir)
		first = ds.describe(0)['ids']
		for i in range(min(8, len(ds.names))):
			ds.describe(i)
		check('cache stays within the bound', len(S._FILE_CACHE) <= 4,
			f'{len(S._FILE_CACHE)} entries, limit 4')
		# index 0's files are long evicted by now; re-describing must reproduce them exactly
		check('content still correct after eviction', ds.describe(0)['ids'] == first)
		# and _set_cache_limit must never shrink a shared cache out from under another dataset
		S._set_cache_limit(2)
		check('_set_cache_limit does not lower the bound', S._CACHE_LIMIT == 4, str(S._CACHE_LIMIT))
	finally:
		S._CACHE_LIMIT = saved_limit
		S._FILE_CACHE.clear()
		S._FILE_CACHE.update(saved_cache)


def check_handle_bound (zip_root, source_dir, target_dir):
	'''6b. Open archives are file descriptors, so they need a bound of their own.

	The parsed-file LRU does not limit handles: a 256-shard corpus read by train and val wants 512 fds
	against a 1024 soft limit on the training box. Closing the least-recently-used archive trades a
	reopen (5.4 ms vs 0.11 ms held) for an fd, so what must hold is that eviction happens AND that a
	reopened archive still yields identical content.
	'''
	print('\n== 6b. handle bound')
	saved = S._ZipSource.MAX_HANDLES
	saved_cache = dict(S._FILE_CACHE)
	try:
		S._ZipSource.MAX_HANDLES = 3
		# a warm parse cache would serve every describe() without ever opening an archive, so the bound
		# would look respected because nothing happened. Start cold.
		S._FILE_CACHE.clear()
		ds = feeder(zip_root, source_dir, target_dir)
		first = ds.describe(0)['ids']
		for i in range(min(20, len(ds.names))):
			ds.describe(i)
		held = len(ds.source._handles)
		check('archives were actually opened', held > 0, f'{held} open')
		check('handles stay within the bound', held <= 3, f'{held} open, bound 3')
		# clear the parse cache so describe(0) must genuinely reopen its (evicted) archive
		S._FILE_CACHE.clear()
		check('content identical through a reopened archive', ds.describe(0)['ids'] == first)
	finally:
		S._ZipSource.MAX_HANDLES = saved
		S._FILE_CACHE.clear()
		S._FILE_CACHE.update(saved_cache)


def check_detection (dir_root, zip_root):
	'''7. Auto-detection is what keeps existing configs working, so it needs its own check.'''
	print('\n== 7. source detection')
	check('a directory root gives _DirSource', isinstance(_make_source(dir_root), _DirSource))
	check('a packed root gives _ZipSource', isinstance(_make_source(zip_root), _ZipSource))
	check('detect() is false for a loose corpus', not _ZipSource.detect(dir_root))
	check('detect() is true for a packed corpus', _ZipSource.detect(zip_root))
	# explicit packed= must override, so a corpus can be forced either way
	check('packed=False forces _DirSource', isinstance(_make_source(dir_root, False), _DirSource))
	# distinct cache keys, or the two roots would share parsed files and the equivalence check
	# above would compare a file against itself
	check('the two sources have distinct cache keys',
		_make_source(dir_root).key != _make_source(zip_root).key)


def main ():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--dir-root', required=True, help='loose corpus root')
	ap.add_argument('--zip-root', required=True, help='packed corpus root (<XX>.zip + index.json)')
	ap.add_argument('--source-dir', default='midi-seq2-irregular')
	ap.add_argument('--target-dir', default='midi-seq2-score')
	ap.add_argument('--limit', type=int, default=0,
		help='compare only the first N samples (0 = all); the equivalence loop parses every file, '
			'which is ~5.6 ms each')
	args = ap.parse_args()

	print('=' * 78)
	print(f'dir {args.dir_root}\nzip {args.zip_root}')
	print(f'pairing {args.source_dir} -> {args.target_dir}')
	print('=' * 78)

	manifest = check_manifest(args.zip_root)
	arms = manifest['arms'] if manifest else [args.source_dir, args.target_dir]
	check_pairs(args.zip_root, manifest, arms)

	dir_ds = feeder(args.dir_root, args.source_dir, args.target_dir)
	zip_ds = feeder(args.zip_root, args.source_dir, args.target_dir)
	check('dir root read as loose files', isinstance(dir_ds.source, _DirSource))
	check('zip root read as archives', isinstance(zip_ds.source, _ZipSource))

	check_names(dir_ds, zip_ds, subset=dir_ds.names != zip_ds.names)
	check_equivalence(dir_ds, zip_ds, args.limit)
	check_collate(dir_ds, zip_ds)
	check_cache_bound(args.zip_root, args.source_dir, args.target_dir)
	check_handle_bound(args.zip_root, args.source_dir, args.target_dir)
	check_detection(args.dir_root, args.zip_root)

	print('\n' + '=' * 78)
	if failures:
		print(f'seq2seq2 archive: {len(failures)} FAILED')
		for f in failures:
			print(f'  - {f}')
		sys.exit(1)
	print('seq2seq2 archive: all checks ok')


if __name__ == '__main__':
	main()
