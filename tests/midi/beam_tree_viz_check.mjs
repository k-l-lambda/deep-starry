/* Run the viewer's OWN functions against a real dump, in node.
   Extracted from the HTML by regex rather than retyped, so a divergence between what is tested and
   what ships is not possible: if the extraction fails, the check fails loudly. */
import { readFileSync } from 'fs';

/* usage: node tests/midi/beam_tree_viz_check.mjs <beamtree.json>
   Checks the viewer's data model against a real dump. The functions under test are EXTRACTED from
   the HTML, not retyped, so this cannot drift away from what the page actually runs. */

const html = readFileSync('tests/midi/beam_tree_viz.html', 'utf8');
const grab = name => {
	const m = html.match(new RegExp(`\\nfunction ${name} \\([^)]*\\) \\{[\\s\\S]*?\\n\\}`, ''));
	if (!m) throw new Error(`could not extract ${name}() from the HTML`);
	return m[0];
};
const src = [grab('winningPath'), grab('ancestry'), grab('pathNotes')].join('\n');
const DATA = JSON.parse(readFileSync(process.argv[2], 'utf8'));
let WIN = 0;
const win = () => DATA.windows[WIN];
const { winningPath, ancestry, pathNotes } = new Function('DATA', 'win',
	src + '\nreturn {winningPath, ancestry, pathNotes};')(DATA, win);

// The whole script block must at least PARSE. new Function compiles without executing, so a syntax
// error surfaces here instead of as a blank page in a browser.
const block = html.match(/<script>\n'use strict';\n([\s\S]*?)\n<\/script>/);
if (!block) throw new Error('could not find the page script block');
try { new Function(block[1]); }
catch (err) { console.log('FAIL page script does not parse: ' + err.message); process.exit(1); }
console.log('ok   page script parses');

let fails = 0;
const ok = (cond, msg) => { console.log((cond ? 'ok   ' : 'FAIL ') + msg); if (!cond) fails++; };

const w = win();
const path = winningPath();
ok(path.length > 0, `winningPath is non-empty: ${path.length} nodes`);
ok(w.best_uid != null, `window records best_uid: ${w.best_uid}`);

// the path must be contiguous in position, one node per decode position, no gaps
const depths = path.map((c, i) => i);
let contiguous = true, prevParent = null;
for (let i = 1; i < path.length; i++)
	if (path[i].parent !== path[i - 1].uid) contiguous = false;
ok(contiguous, 'each winning node\'s parent is the previous winning node (no broken links)');
ok(path.length === w.positions.length,
	`winning path covers every position: ${path.length} == ${w.positions.length}`);
ok(path.every(c => c.kept), 'every node on the winning path was kept');

// the path's tokens must equal the emitted output for the window
const notes = pathNotes(path);
ok(notes.length > 0, `winning path produced note_on events: ${notes.length}`);
ok(notes.every(n => n.align), 'every path note carries an alignment verdict');
const matched = notes.filter(n => n.align.src !== null).length;
console.log(`     of ${notes.length} notes on the path, ${matched} matched a source note`);

// ancestry of a CUT node must end at a kept node, and be one shorter than its position
const cut = w.positions.flatMap(p => p.candidates.filter(c => !c.kept && !c.eos)
	.map(c => ({ c, pos: p })));
ok(cut.length > 0, `dump contains cut candidates: ${cut.length}`);
const sample = cut[Math.floor(cut.length / 2)];
const anc = ancestry(sample.c);
ok(anc.length === sample.pos.position,
	`a cut node's ancestry length equals its position: ${anc.length} == ${sample.pos.position}`);
ok(anc.every(c => c.kept), 'a cut node\'s ancestors were all kept (it hangs off a live lineage)');

// elapse detection, the branch the onset view keys its vertical rule on
const re = /^E[0-9a-f]+$/i;
const elapses = w.positions.flatMap(p => p.candidates.filter(c => re.test(c.token)));
ok(elapses.length > 0, `elapse candidates detected by the viewer's regex: ${elapses.length}`);
ok(elapses.every(c => c.align === null),
	'no elapse candidate carries an alignment verdict (it closes no note)');
const pitched = w.positions.flatMap(p => p.candidates.filter(c => c.align !== null));
ok(pitched.every(c => /^#[0-9a-f]+$/i.test(c.token)),
	'every candidate WITH a verdict is a #pitch token');

// tick monotonicity along the path: the clock may not go backwards
let mono = true;
for (let i = 1; i < path.length; i++) if (path[i].tick < path[i - 1].tick) mono = false;
ok(mono, 'tick never decreases along the winning path');

// --- the shared-scale contract -------------------------------------------------------
// Both lanes are plotted on ONE scale in either unit, so every candidate needs a finite coordinate
// in BOTH. An elapse has no softIndex of its own (softIndex only advances when a note happens), so
// the dump carries a provisional `si` -- without it an elapse's rule has nowhere to land.
const cands = w.positions.flatMap(p => p.candidates);
ok(cands.every(c => typeof c.si === 'number' && isFinite(c.si)),
	`every candidate carries a finite si: ${cands.length} candidates`);
ok(cands.every(c => typeof c.tick === 'number' && isFinite(c.tick)),
	'every candidate carries a finite tick');
const drift = cands.filter(c => c.align)
	.map(c => Math.abs(c.si - c.align.softIndex));
ok(drift.every(d => d < 1e-4),
	`a pitch candidate's provisional si equals its align.softIndex (max delta ${
		drift.length ? Math.max(...drift).toExponential(1) : 0})`);
let siMono = true;
for (let i = 1; i < path.length; i++) if (path[i].si < path[i - 1].si - 1e-9) siMono = false;
ok(siMono, 'si never decreases along the winning path');
const ssi = DATA.source.map(e => e.softIndex);
let srcMono = true;
for (let i = 1; i < ssi.length; i++) if (ssi[i] < ssi[i - 1] - 1e-9) srcMono = false;
ok(srcMono, `source softIndex is non-decreasing (0..${ssi[ssi.length - 1]})`);
ok(DATA.source.every(e => typeof e.softIndex === 'number'),
	'every source event carries softIndex, so the source lane can be drawn in either unit');

console.log(fails ? `\n${fails} check(s) FAILED` : '\nall checks passed');
process.exit(fails ? 1 : 0);
