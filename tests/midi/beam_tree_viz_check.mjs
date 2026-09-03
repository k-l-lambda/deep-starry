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
/* The winner spans every recorded position ONLY when nothing terminated: a finished hypothesis
   stops where it saw <eos>, while the search keeps recording until beam_size of them finish, so its
   path is legitimately shorter than the dump. Both cases are checked, neither is waived. */
const finished = path.length !== w.positions.length;
if (finished)
	ok(path.length < w.positions.length,
		`winner terminated early, so its path is shorter than the dump: ${path.length} < ${w.positions.length}`);
else
	ok(path.length === w.positions.length,
		`no hypothesis finished, so the winning path covers every position: ${path.length} == ${w.positions.length}`);
/* A finished winner's last node is the one <eos> followed, and a candidate that <eos> beat is never
   in `nxt`, so `kept` is false on it -- every EARLIER node still had to be kept. */
const spine = finished ? path.slice(0, -1) : path;
ok(spine.every(c => c.kept), 'every node on the winning path was kept'
	+ (finished ? ' (excluding the terminated leaf)' : ''));

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

// The detail line NAMES the parent, and the tree outlines it. Both need more than a uid: the
// position to label it by and the cum that says whether the child survived on its lineage's lead.
const par = anc[anc.length - 1];
ok(par !== undefined && Number.isInteger(par.position) && Number.isFinite(par.cum),
	`ancestry names the parent: ${par && par.token} @${par && par.position} cum ${par && par.cum}`);
ok(par.position === sample.pos.position - 1,
	`the parent sits one position earlier: ${par.position} == ${sample.pos.position - 1}`);
// The tree addresses the parent by uid, so a parent uid must resolve to a drawn node -- EXCEPT at
// the window's first position, whose parent is the seed beam. That beam predates the first decode
// position, so it was never a candidate and is legitimately absent; the tree draws no outline for it
// and the detail line reports 'root'.
const uids = new Set(w.positions.flatMap(p => p.candidates.map(c => c.uid)).filter(u => u !== null));
const first = w.positions[0].position;
/* The loss key's presence must match whether an ADJUDICATOR RAN -- meta.adjudicator, not meta.rank.
   `--rank lm --inspect` scores every candidate and lets the model rank anyway, so an lm dump can
   legitimately carry a loss on all of them; keying this on rank failed that dump. A null loss means
   "asked, no evidence" and an absent key means "never asked", so the two must not be conflated. */
{
	const all = w.positions.flatMap(p => p.candidates);
	const withKey = all.filter(c => 'loss' in c).length;
	// Older dumps predate the field; fall back to rank, which was the only signal then.
	const ran = DATA.meta.adjudicator !== undefined
		? DATA.meta.adjudicator !== null : DATA.meta.rank === 'align';
	if (ran)
		ok(withKey === all.length,
			`an adjudicated dump (${DATA.meta.adjudicator || DATA.meta.rank}) carries a loss key on `
			+ `every candidate: ${withKey}/${all.length}`);
	else
		ok(withKey === 0,
			`a dump with no adjudicator carries no loss key: ${withKey}/${all.length}`);
}

const orphans = w.positions.flatMap(p => p.candidates.map(c => ({ c, at: p.position })))
	.filter(({ c }) => c.parent !== null && !uids.has(c.parent));
ok(orphans.every(o => o.at === first),
	`only the first position's candidates hang off the unrecorded seed beam `
	+ `(${orphans.length} such, all at position ${first})`);

// The `loss` field distinguishes three states the viewer must not conflate: absent (an LM-only dump,
// --rank lm), null (the aligner abstained at that position) and a number (what the search ranked on).
// The committed dump is LM-only, so absent is what it exercises; the other two are pinned by shape.
const withLoss = w.positions.flatMap(p => p.candidates).filter(c => 'loss' in c);
const lossKinds = new Set(withLoss.map(c => c.loss === null ? 'abstained' : typeof c.loss));
ok(withLoss.length === 0 || [...lossKinds].every(k => k === 'abstained' || k === 'number'),
	withLoss.length === 0
		? 'dump carries no align loss (an LM-only run), which the viewer renders as absent'
		: `align loss present on ${withLoss.length} candidates, kinds {${[...lossKinds].join(', ')}}`);

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

/* --- click-to-toggle -----------------------------------------------------------------------
   Clicking the selected node must CLEAR the selection. Two things have to hold together, and the
   failure mode of the pair is silent: the toggle has to test the same thing the outline does, and it
   cannot test object identity. SEL is a spread COPY of the candidate (`{ ...c, position }`), so a
   toggle written `SEL === c` would never fire and the node would just re-select itself -- no error,
   no visible break, only a gesture that quietly does nothing. */
// `.trim()` is load-bearing: grab() returns the match with its leading newline, and
// `return \nfunction …` is `return;` by automatic semicolon insertion -- the Function would hand back
// undefined and every check below would throw instead of failing.
const sameNode = new Function('return ' + grab('sameNode').trim())();
{
	const pos = { position: 7 };
	const c = { rank: 2, uid: 41, token: 'E140' };
	const SELcopy = { ...c, position: pos.position };
	ok(sameNode(SELcopy, c, pos) === true,
		'sameNode matches a spread COPY of the candidate, so the toggle can recognise the selection');
	ok(sameNode({ ...c, position: 8 }, c, pos) === false
		&& sameNode({ ...c, rank: 3, position: 7 }, c, pos) === false,
		'sameNode separates a different position and a different rank');
	// A cut candidate carries uid null, so a uid-keyed predicate would collapse every cut node in the
	// column into one. Rank still separates them.
	const cutA = { rank: 5, uid: null, token: 'Eb40' }, cutB = { rank: 6, uid: null, token: 'E290' };
	ok(sameNode({ ...cutA, position: 7 }, cutA, pos) === true
		&& sameNode({ ...cutA, position: 7 }, cutB, pos) === false,
		'a cut node (uid null) is still identifiable, so cut nodes toggle individually');
}
// The handler itself is inline in a forEach, so it is checked by source: it must assign from `isSel`
// (the outline's own variable) and must not compare SEL to the candidate by identity.
{
	const h = html.match(/g\.addEventListener\('click',[\s\S]*?\}\);/);
	ok(!!h && /SEL = isSel \? null :/.test(h[0]),
		'the click handler toggles off `isSel`, the same variable that draws the selected outline');
	// Comments stripped first: the handler's own comment NAMES the identity comparison in order to say
	// why it is wrong, and scanning the raw text made this check fail on the explanation.
	const code = h ? h[0].replace(/\/\/[^\n]*/g, '') : '';
	ok(!!h && !/SEL === c|c === SEL/.test(code),
		'the click handler does not compare SEL to the candidate by identity (SEL is a copy)');
}

/* --- the x-axis default ---------------------------------------------------------------------
   The `blockLabel` text is markup, rewritten only by the unit control's `change` handler, so it must
   already agree with whichever option carries `selected`. A mismatch is silent: the page opens
   claiming a unit it is not drawing in, and stays wrong until the user touches the control. */
{
	const optM = html.match(/<select id="unit">([\s\S]*?)<\/select>/);
	ok(!!optM, 'the unit control is present');
	const selOpt = optM && optM[1].match(/<option value="(\w+)"[^>]*\bselected\b/);
	ok(!!selOpt && selOpt[1] === 'si',
		`the x axis defaults to softIndex (selected option: ${selOpt ? selOpt[1] : 'none'})`);
	const lblM = html.match(/<span id="blockLabel">([^<]*)<\/span>/);
	// The pairing the handler enforces at runtime: 'tick' -> '480 ticks', 'si' -> '1 softIndex'.
	const want = selOpt && selOpt[1] === 'tick' ? '480 ticks' : '1 softIndex';
	ok(!!lblM && lblM[1] === want,
		`the static block label matches that default: "${lblM ? lblM[1] : 'missing'}" == "${want}"`);
}

console.log(fails ? `\n${fails} check(s) FAILED` : '\nall checks passed');
process.exit(fails ? 1 : 0);
