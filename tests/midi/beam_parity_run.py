'''End-to-end parity: `translateMidiseq2Beam.py --beam 1` vs `translateMidiseq2.py`, same checkpoint.

The stub-model checks in beam_parity_check.py pin the SEARCH; this pins the INTEGRATION. They fail
differently: a stub cannot catch a wrong pos_style being threaded through, a primer that is seeded
from the wrong half of the prefix, or an EncDec memory that is expanded along the wrong axis. Those
only appear once a real model is producing real logits over a real sliding window.

Byte-identity is the bar, not similarity. `--beam 1` returns super().generate(...), so the two runs
execute the same lines; anything less than identical output means the two scripts disagree about the
window, the positions or the primer, and that has to be fixed before a width-4 result means anything.

Run:
  python tests/midi/beam_parity_run.py --run <run_dir> --input a.midiseq2.txt --max-steps 3
  python tests/midi/beam_parity_run.py --run <run_dir> --input a.txt --beam 4   # also time width 4
'''

import argparse
import hashlib
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(REPO_ROOT, 'tools', 'midi')


def say (*args):
	'''Print and flush. The legs take tens of seconds each, and a buffered driver looks identical to
	a hung one from outside -- which cost a round of guessing the first time.'''
	print(*args, flush=True)


def digest (path):
	with open(path, 'rb') as f:
		return hashlib.sha256(f.read()).hexdigest()[:16]


def run (cmd, log):
	start = time.time()
	with open(log, 'w', encoding='utf-8') as f:
		proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
	return proc.returncode, time.time() - start


def main ():
	ap = argparse.ArgumentParser(description='Byte-identity parity between the greedy and beam tools.')
	ap.add_argument('--run', required=True)
	ap.add_argument('--input', required=True)
	ap.add_argument('--out-dir', default='/tmp/beam_parity')
	ap.add_argument('--max-steps', type=int, default=3,
		help='windows per run (default 3). Parity is per-step, so a few steps settle it; the whole '
			'file only costs time')
	ap.add_argument('--src-window', type=int, default=640)
	ap.add_argument('--max-token', type=int, default=2048)
	ap.add_argument('--prime-window', type=int, default=2048)
	ap.add_argument('--threads', type=int, default=8)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--beam', type=int, default=0,
		help='also run this width and report its cost against greedy (0 = parity only). No output '
			'comparison: a wider beam is SUPPOSED to differ, and that difference is what the '
			'accuracy harness measures, not this script')
	args = ap.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	common = ['--run', args.run, '--input', args.input, '--max-steps', str(args.max_steps),
		'--src-window', str(args.src_window), '--max-token', str(args.max_token),
		'--prime-window', str(args.prime_window), '--threads', str(args.threads),
		'--device', args.device]

	greedy_out = os.path.join(args.out_dir, 'greedy.txt')
	beam1_out = os.path.join(args.out_dir, 'beam1.txt')

	say(f'[1/2] greedy: translateMidiseq2.py')
	rc_g, t_g = run([sys.executable, os.path.join(TOOLS, 'translateMidiseq2.py'),
		'--output', greedy_out] + common, os.path.join(args.out_dir, 'greedy.log'))
	say(f'      rc {rc_g}, {t_g:.1f}s')

	say(f'[2/2] beam 1: translateMidiseq2Beam.py --beam 1')
	rc_b, t_b = run([sys.executable, os.path.join(TOOLS, 'translateMidiseq2Beam.py'),
		'--beam', '1', '--output', beam1_out] + common, os.path.join(args.out_dir, 'beam1.log'))
	say(f'      rc {rc_b}, {t_b:.1f}s')

	if rc_g or rc_b:
		say(f'\nFAIL a run did not complete (rc {rc_g}/{rc_b}); see {args.out_dir}/*.log')
		return 1

	d_g, d_b = digest(greedy_out), digest(beam1_out)
	same = d_g == d_b
	say(f'\n{"ok  " if same else "FAIL"} byte identity: greedy {d_g}  beam1 {d_b}')
	if not same:
		diff = subprocess.run(['diff', '-u', greedy_out, beam1_out], capture_output=True, text=True)
		lines = diff.stdout.splitlines()
		say(f'     {len(lines)} diff lines; first 20:')
		for line in lines[:20]:
			say(f'     {line}')

	if args.beam > 1:
		beam_out = os.path.join(args.out_dir, f'beam{args.beam}.txt')
		say(f'\n[+] beam {args.beam} cost')
		rc, t = run([sys.executable, os.path.join(TOOLS, 'translateMidiseq2Beam.py'),
			'--beam', str(args.beam), '--output', beam_out] + common,
			os.path.join(args.out_dir, f'beam{args.beam}.log'))
		if rc:
			say(f'    FAIL rc {rc}; see {args.out_dir}/beam{args.beam}.log')
			return 1
		say(f'    rc {rc}, {t:.1f}s  ({t / max(t_g, 1e-6):.2f}x greedy)')
		with open(os.path.join(args.out_dir, f'beam{args.beam}.log'), encoding='utf-8') as f:
			for line in f:
				if line.startswith('[beam]') or line.startswith('[out]'):
					say(f'    {line.rstrip()}')
		# Differing from greedy is the POINT of a wider beam; identical output would mean the extra
		# width bought nothing, which is worth saying out loud rather than reading as success.
		if digest(beam_out) == d_g:
			say(f'    note: beam {args.beam} output is IDENTICAL to greedy -- the width changed '
				'nothing on this file')

	return 0 if same else 1


if __name__ == '__main__':
	sys.exit(main())
