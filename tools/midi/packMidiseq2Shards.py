'''Pack a loose midiseq2 corpus into per-shard zip archives for the Seq2Seq2 feeder.

    <root>/<arm>/<id>.midiseq2.txt          ->      <out>/<XX>.zip   holding <arm>/<id>.midiseq2.txt
                                                    <out>/index.json

A shard is the first two characters of the filename, so a sample's archive is computable from its name
alone and the feeder needs the manifest only to enumerate. BOTH arms go into the same shard archive,
which keeps a pair in one file (one handle serves both halves of a sample).

Why bother: midiseq2 is repetitive text and compresses hard — measured 0.067 on the score arm and 0.144
on the irregular arm, taking nota1m's 124 G of loose files to ~15 G, and replacing ~1.5M inodes with
~256 archives. It is free at training time: parsing dominates the feeder's per-file cost (5.6 ms) and a
decompressed read is 0.1-0.4 ms of it.

Only the basename INTERSECTION of the arms is packed. The feeder intersects anyway, so an unpaired file
would be dead weight; doing it here makes each archive self-consistent.

In packed mode the feeder's `source_dir`/`target_dir` are ENTRY PREFIXES rather than directories, so
`--publish-as` decouples the name a config uses from the directory the corpus happened to be built in
— pack a staging `midi-seq2-irregular2/` and publish it as `midi-seq2-irregular`. Note the corollary:
because the prefix is baked into every entry, renaming an arm after the fact means repacking, which a
`mv` on the loose corpus cannot substitute for.

Resumable per shard: an existing `<XX>.zip` is skipped unless --force, so an interrupted run continues.
The manifest is rewritten from whatever archives exist at the end, so it always describes the output.

Example — nota1m, whose irregular arm was built into a suffixed staging directory but is published
without the suffix, so no config or downstream host ever names it `midi-seq2-irregular2`:
    python tools/midi/packMidiseq2Shards.py \\
        --root /models/nota1m --arms midi-seq2-irregular2 midi-seq2-score \\
        --publish-as midi-seq2-irregular midi-seq2-score \\
        --out /models/nota1m-packed --workers 8
'''

import argparse
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from multiprocessing import Pool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

SHARD_CHARS = 2
MANIFEST = 'index.json'


def shard_of (name):
	return name[:SHARD_CHARS]


def scan (root, arms):
	'''Group the arms' shared basenames by shard. Returns (shards, stats).

	Enumeration is one listdir per arm — the expensive part of the whole tool at 1.5M files, and the
	reason the feeder wants an archive in the first place.
	'''
	listings = {}
	for arm in arms:
		path = os.path.join(root, arm)
		if not os.path.isdir(path):
			raise SystemExit(f'no such arm directory: {path}')
		listings[arm] = {n for n in os.listdir(path) if n.endswith('.txt')}
		print(f'  {arm}: {len(listings[arm])} files')

	shared = set.intersection(*listings.values())
	shards = defaultdict(list)
	for name in shared:
		shards[shard_of(name)].append(name)
	for names in shards.values():
		names.sort()

	stats = {arm: len(names) for arm, names in listings.items()}
	stats['shared'] = len(shared)
	# Unpaired counts are worth printing rather than silently dropping: on nota1m the 43k
	# irregular-only files are the samples whose measure mapping the score arm rejected.
	for arm, names in listings.items():
		stats[f'{arm}_only'] = len(names - shared)
	return dict(shards), stats


def pack_shard (job):
	'''One shard -> one zip. Runs in a worker, so it takes and returns only picklable data.'''
	root, arms, publish, out, shard, names, force, level = job
	path = os.path.join(out, f'{shard}.zip')
	if os.path.isfile(path) and not force:
		return dict(shard=shard, skipped=True, entries=0, raw=0, size=os.path.getsize(path), seconds=0.0)

	t0 = time.time()
	raw = 0
	# Write to a temp name and rename, so an interrupted run leaves no half archive that the next run
	# would then skip as complete.
	tmp = path + '.part'
	with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=level) as z:
		for arm in arms:
			# The entry prefix is the PUBLISHED name, which need not be the source directory name. In
			# packed mode the feeder's source_dir/target_dir are entry prefixes, not directories
			# (seq2seq2.py _ZipSource.read opens f'{arm}/{name}'), so this is what lets a staging
			# directory carry a build-time name while configs name the arm however they should.
			for name in names:
				src = os.path.join(root, arm, name)
				raw += os.path.getsize(src)
				z.write(src, f'{publish[arm]}/{name}')
	os.replace(tmp, path)
	return dict(shard=shard, skipped=False, entries=len(names) * len(arms), raw=raw,
		size=os.path.getsize(path), seconds=time.time() - t0)


def write_manifest (out, arms, shards):
	'''The manifest the feeder enumerates from.

	JSON, not YAML, deliberately: every other packed dataset here ships index.yaml, but yaml.safe_load
	of a 200k-name index measured 16.4 s against 50 ms for JSON — a 330x difference that would cost
	minutes per process start at this corpus's ~1.5M names.

	Rebuilt from the archives actually on disk rather than from the scan, so it can never promise a
	shard that a partial run did not write.

	`shards` must be the FULL scan, not the subset this run packed: an incremental run passing only its
	own batch would emit a manifest describing that batch alone, dropping every previously packed shard
	from the feeder's view while their archives sit on disk unreferenced.
	'''
	present = {}
	for entry in sorted(os.listdir(out)):
		if not entry.endswith('.zip'):
			continue
		shard = entry[:-4]
		if shard in shards:
			present[shard] = shards[shard]
	manifest = dict(arms=list(arms), shard_chars=SHARD_CHARS, shards=present,
		samples=sum(len(v) for v in present.values()))
	path = os.path.join(out, MANIFEST)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(manifest, f)
	return manifest, path


def main ():
	ap = argparse.ArgumentParser(description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--root', required=True, help='corpus root holding the arm directories')
	ap.add_argument('--arms', nargs='+', required=True,
		help='arm directory names to pack (e.g. midi-seq2-irregular2 midi-seq2-score)')
	ap.add_argument('--publish-as', nargs='+', default=None, metavar='NAME',
		help='entry prefix to write for each --arms entry, in the same order. In packed mode the '
			'feeder\'s source_dir/target_dir ARE the entry prefixes, so this decouples the name '
			'configs use from the directory the corpus was built in (e.g. pack a staging '
			'midi-seq2-irregular2/ and publish it as midi-seq2-irregular). Defaults to --arms.')
	ap.add_argument('--out', required=True, help='destination dir for <XX>.zip + index.json')
	ap.add_argument('--only', nargs='+', default=None,
		help='pack just these shards (e.g. --only 00 01) — for subset tests')
	ap.add_argument('--workers', type=int, default=4,
		help='shards packed in parallel (default 4; measured ~16 s/shard single-threaded, so raise it '
			'only when the box is idle)')
	ap.add_argument('--level', type=int, default=6, help='deflate level (default 6)')
	ap.add_argument('--force', action='store_true', help='repack shards whose zip already exists')
	args = ap.parse_args()

	published = args.publish_as if args.publish_as is not None else args.arms
	if len(published) != len(args.arms):
		raise SystemExit(f'--publish-as takes one name per --arms entry: '
			f'{len(args.arms)} arms, {len(published)} published names')
	if len(set(published)) != len(published):
		raise SystemExit(f'--publish-as names must be distinct, got {published}')
	publish = dict(zip(args.arms, published))

	print(f'[scan] {args.root}')
	shards, stats = scan(args.root, args.arms)
	for arm, name in publish.items():
		if arm != name:
			print(f'  publishing {arm} as {name}')
	print(f'  shared basenames: {stats["shared"]} across {len(shards)} shards')
	for arm in args.arms:
		only = stats[f'{arm}_only']
		if only:
			print(f'  {arm}-only (not packed): {only}')

	# Kept whole for the manifest: `shards` below may be narrowed to this run's batch, but the manifest
	# must describe every archive on disk (see write_manifest).
	all_shards = shards
	if args.only:
		missing = [s for s in args.only if s not in shards]
		if missing:
			raise SystemExit(f'requested shards not present in the corpus: {missing}')
		shards = {s: shards[s] for s in args.only}
		print(f'  --only: {len(shards)} shard(s)')

	os.makedirs(args.out, exist_ok=True)
	jobs = [(args.root, args.arms, publish, args.out, shard, names, args.force, args.level)
		for shard, names in sorted(shards.items())]

	t0 = time.time()
	results = []
	if args.workers > 1 and len(jobs) > 1:
		with Pool(min(args.workers, len(jobs))) as pool:
			for r in pool.imap_unordered(pack_shard, jobs):
				results.append(r)
				print(f'  [{len(results)}/{len(jobs)}] {r["shard"]} '
					+ ('skipped' if r['skipped'] else
						f'{r["entries"]} entries {r["size"]/1e6:.1f} MB '
						f'ratio {r["size"]/max(1, r["raw"]):.3f} {r["seconds"]:.0f}s'))
	else:
		for job in jobs:
			r = pack_shard(job)
			results.append(r)
			print(f'  [{len(results)}/{len(jobs)}] {r["shard"]} '
				+ ('skipped' if r['skipped'] else
					f'{r["entries"]} entries {r["size"]/1e6:.1f} MB '
					f'ratio {r["size"]/max(1, r["raw"]):.3f} {r["seconds"]:.0f}s'))

	# PUBLISHED names, not source directory names: the manifest's `arms` is what tells a reader which
	# entry prefixes the archives actually contain, so recording the staging names would misdescribe them.
	manifest, path = write_manifest(args.out, published, all_shards)
	raw = sum(r['raw'] for r in results)
	size = sum(r['size'] for r in results)
	packed = sum(1 for r in results if not r['skipped'])
	print(f'\n[done] {packed} packed, {len(results) - packed} skipped, '
		f'{manifest["samples"]} samples in {len(manifest["shards"])} shards, {time.time() - t0:.0f}s')
	if raw:
		print(f'  {raw/1e9:.2f} G raw -> {size/1e9:.2f} G ({size/raw:.3f})')
	print(f'  manifest: {path}')


if __name__ == '__main__':
	main()
