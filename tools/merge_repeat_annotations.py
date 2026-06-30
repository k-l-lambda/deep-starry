import json, csv, re
from pathlib import Path

base = Path('/home/camus/data/scores/fmenu-midi-matches-filtered')
ann = json.loads((base / 'repeat_annotations.json').read_text())

# index annotations by piece
by_piece = {a['piece']: a for a in ann}

csv_path = base / 'summary.csv'
with csv_path.open(newline='') as f:
    rows = list(csv.DictReader(f))
fields = list(rows[0].keys())

new_cols = ['scoreRepeatExpansion', 'repeatKind', 'repeatPrimaryCause', 'repeatConfidence', 'repeatEvidence']
for c in new_cols:
    if c not in fields:
        fields.append(c)

hit = 0
for r in rows:
    piece = Path(r['pieceDir']).name
    a = by_piece.get(piece)
    if a:
        r['scoreRepeatExpansion'] = 'true' if a.get('scoreHasRepeatedSection') else 'false'
        r['repeatKind'] = a.get('repeatKind', '')
        r['repeatPrimaryCause'] = a.get('primaryCause', '')
        r['repeatConfidence'] = a.get('confidence', '')
        r['repeatEvidence'] = a.get('evidence', '')
        if a.get('scoreHasRepeatedSection'):
            hit += 1
    else:
        # not studied (scoreUnmatchStrict < 0.3) -> leave blank / false
        r.setdefault('scoreRepeatExpansion', '')
        for c in new_cols:
            r.setdefault(c, '')

with csv_path.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, '') for k in fields})

print('rows', len(rows), 'annotated', len(ann), 'scoreRepeatExpansion=true', hit)
print('new columns:', new_cols)
