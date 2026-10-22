'''Pack a midiseq2 arm + Lilylet files + measure maps into the InterleavedSeq2 packed layout.

    <packages>/<XX>.zip  <arm>/<id>.midiseq2.txt  ┐
    <lyl>/<XX>/<YY>/<id>.lyl                      ├─>  <out>/<XX>.zip   <arm>/<id>.midiseq2.txt
    <measures>/measures-<XX>.json  {id: record}   ┘                     lyl/<id>.lyl
                                                                        measures/<id>.json
                                                       <out>/index.json  layout "interleaved"

`<packages>` is an existing packMidiseq2Shards.py archive set (e.g. nota1m's
packages-irregular-e80bcc66-hi045-pp); only the one midiseq2 arm named by --arm is carried, copied
byte for byte. `measures/<id>.json` is the bare `source_measure` list of the record's played measures
— all the feeder needs — rather than the whole record, whose tick and path fields it never reads.

Every pair is VALIDATED here, once, so the feeder can trust the manifest and does no per-pair work at
init (the loose layout's `_init_mixed` reads every measures.json record up front, which does not scale
to 668k pairs). A pair is rejected, and counted by reason in the manifest, when:

    no_record / no_lyl        a measures record or a .lyl file is missing
    not_ok                    record `ok` is false
    repeat_unaligned          `repeat_aligned` is false (no honest map back to notation)
    bad_rows                  rows not contiguous from 1, or a source_measure that is not an int >= 1
    lyl_short                 the map names a Lilylet measure past the body's last measure
    marks_mismatch            the midiseq2 @measure marks are not exactly 1..len(rows)

which are the same conditions `Seq2Seq2._init_mixed` / `_mixed_lilylet_document` / `_mixed_midi`
raise on, so a packed pair can never fail them at training time.

Resumable per shard: an existing `<XX>.zip` with its `<XX>.json` sidecar is skipped unless --force. Each
archive is written to a temporary name and renamed only when complete. The manifest is rebuilt from the
sidecars at the end, so it always describes exactly the archives present.

Example — nota1m irregular -> Lilylet, on 196_2:
    python tools/midi/packInterleavedLilylet.py \\
        --packages /home/claude/datasets/midi/nota1m/packages-irregular-e80bcc66-hi045-pp \\
        --arm midi-seq2-irregular \\
        --lyl /home/claude/datasets/midi/nota1m/lyl \\
        --measures /home/claude/datasets/midi/nota1m/provenance/metadata \\
        --out /home/claude/datasets/midi/nota1m-interleaved-lyl --workers 32
'''

import argparse
import json
import os
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

MANIFEST = 'index.json'
MIDI_SUFFIX = '.midiseq2.txt'
LYL_ARM = 'lyl'
MEASURES_ARM = 'measures'


def mapping_of (record):
	'''Record -> (source_measure list, None) or (None, reject reason). Mirrors Seq2Seq2._init_mixed.'''
	if not isinstance(record, dict):
		return None, 'no_record'
	if record.get('ok') is False:
		return None, 'not_ok'
	rows = record.get('measures')
	if not isinstance(rows, list) or not rows:
		return None, 'bad_rows'
	mapping = []
	for expected, row in enumerate(rows, 1):
		if not isinstance(row, dict) or row.get('index') != expected:
			return None, 'bad_rows'
		source = row.get('source_measure')
		if source is None and record.get('repeat_aligned') is False:
			return None, 'repeat_unaligned'
		if not isinstance(source, int) or isinstance(source, bool) or source < 1:
			return None, 'bad_rows'
		mapping.append(source)
	if record.get('repeat_aligned') is False:
		return None, 'repeat_unaligned'
	return mapping, None


def measure_marks (text):
	return [int(line.split()[1]) for line in text.splitlines() if line.startswith('@measure')]


def pack_shard (job):
	shard, args = job
	from starry.lilylet.data.patchifier import split_lilylet_document, split_measures

	final = os.path.join(args.out, f'{shard}.zip')
	sidecar = os.path.join(args.out, f'{shard}.json')
	if not args.force and os.path.exists(final) and os.path.exists(sidecar):
		return shard, None
	started = time.time()
	with open(os.path.join(args.measures, f'measures-{shard}.json'), 'r', encoding='utf-8') as f:
		records = json.load(f)
	src = zipfile.ZipFile(os.path.join(args.packages, f'{shard}.zip'), 'r')
	prefix = f'{args.arm}/'
	names = sorted(n[len(prefix):] for n in src.namelist() if n.startswith(prefix) and n.endswith(MIDI_SUFFIX))
	if args.limit:
		names = names[:args.limit]

	rejects = Counter()
	kept = []
	tmp = final + '.tmp'
	with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as out:
		for name in names:
			sid = name[:-len(MIDI_SUFFIX)]
			mapping, reason = mapping_of(records.get(sid))
			if reason:
				rejects[reason] += 1
				continue
			lyl_path = os.path.join(args.lyl, sid[:2], sid[2:4], sid + '.lyl')
			if not os.path.isfile(lyl_path):
				rejects['no_lyl'] += 1
				continue
			with open(lyl_path, 'r', encoding='utf-8') as f:
				lyl_text = f.read()
			_, body = split_lilylet_document(lyl_text)
			if len(split_measures(body)) < max(mapping):
				rejects['lyl_short'] += 1
				continue
			midi = src.read(prefix + name)
			if measure_marks(midi.decode('utf-8')) != list(range(1, len(mapping) + 1)):
				rejects['marks_mismatch'] += 1
				continue
			out.writestr(prefix + name, midi)
			out.writestr(f'{LYL_ARM}/{sid}.lyl', lyl_text)
			out.writestr(f'{MEASURES_ARM}/{sid}.json', json.dumps(mapping, separators=(',', ':')))
			kept.append(sid)
	os.replace(tmp, final)
	with open(sidecar + '.tmp', 'w', encoding='utf-8') as f:
		json.dump(dict(ids=kept, scanned=len(names), rejects=dict(rejects)), f)
	os.replace(sidecar + '.tmp', sidecar)
	return shard, dict(kept=len(kept), scanned=len(names), rejects=dict(rejects), seconds=round(time.time() - started, 1))


def main ():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--packages', required=True, help='source packMidiseq2Shards archive dir')
	ap.add_argument('--arm', required=True, help='midiseq2 arm to carry, e.g. midi-seq2-irregular')
	ap.add_argument('--lyl', required=True, help='loose Lilylet root, <lyl>/<XX>/<YY>/<id>.lyl')
	ap.add_argument('--measures', required=True, help='dir of measures-<XX>.json records')
	ap.add_argument('--out', required=True)
	ap.add_argument('--workers', type=int, default=8)
	ap.add_argument('--only', nargs='+', default=None, help='pack just these shards (subset tests)')
	ap.add_argument('--limit', type=int, default=0, help='at most N samples per shard (smoke tests)')
	ap.add_argument('--force', action='store_true')
	args = ap.parse_args()

	with open(os.path.join(args.packages, MANIFEST), 'r', encoding='utf-8') as f:
		source_manifest = json.load(f)
	if args.arm not in source_manifest['arms']:
		raise SystemExit(f'--arm {args.arm!r} not in {source_manifest["arms"]}')
	shards = sorted(args.only or source_manifest['shards'])
	os.makedirs(args.out, exist_ok=True)

	started = time.time()
	with Pool(args.workers) as pool:
		for done, (shard, info) in enumerate(pool.imap_unordered(pack_shard, [(s, args) for s in shards]), 1):
			status = 'skipped (exists)' if info is None else json.dumps(info)
			print(f'[{done}/{len(shards)}] {shard} {status}  elapsed {time.time() - started:.0f}s', flush=True)

	manifest_shards, rejects, scanned = {}, Counter(), 0
	for entry in sorted(os.listdir(args.out)):
		if not (entry.endswith('.json') and len(entry) == 7 and entry != MANIFEST):
			continue
		shard = entry[:2]
		if not os.path.exists(os.path.join(args.out, f'{shard}.zip')):
			continue
		with open(os.path.join(args.out, entry), 'r', encoding='utf-8') as f:
			side = json.load(f)
		manifest_shards[shard] = side['ids']
		rejects.update(side['rejects'])
		scanned += side['scanned']
	manifest = dict(
		layout='interleaved',
		shard_chars=2,
		arms=[args.arm, LYL_ARM, MEASURES_ARM],
		samples=sum(len(v) for v in manifest_shards.values()),
		scanned=scanned,
		rejects=dict(rejects),
		source=dict(packages=os.path.abspath(args.packages), arm=args.arm, lyl=os.path.abspath(args.lyl),
			measures=os.path.abspath(args.measures), limit=args.limit or None),
		shards=manifest_shards,
	)
	with open(os.path.join(args.out, MANIFEST + '.tmp'), 'w', encoding='utf-8') as f:
		json.dump(manifest, f)
	os.replace(os.path.join(args.out, MANIFEST + '.tmp'), os.path.join(args.out, MANIFEST))
	print(f'manifest: {manifest["samples"]} samples of {scanned} scanned in {len(manifest_shards)} shards; '
		f'rejects {dict(rejects)}')


if __name__ == '__main__':
	main()
