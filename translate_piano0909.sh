#!/bin/bash
# Translate piano0909/raw/*.midi with the l16d256 nota1m0909 run.
#
#   raw/<id>.midi  --(midi -> midiseq2, in process, per file)-->  a temp .midiseq2.txt
#                  --(translateMidiseq2.py --annotate-source)-->  rubato/<id>.midiseq2.txt   (source + @measure)
#                                                                 regular/<id>.midiseq2.txt  (translated target)
#
# Both outputs are midiseq2. The midi -> midiseq2 step runs on the spot, just before each
# file's translation, via a PERSISTENT node worker (see midiToSeq2Server.ts): ts-node costs
# ~1.7 s to boot, so one process per file would spend hours doing nothing else.
#
# Usage:
#   ./translate_piano0909.sh --workers 4 --gpus 0,1,2,3 [options]
#   ./translate_piano0909.sh --gpus 5,4 --until "2026-09-16 08:16"
#
#   --workers N        concurrent workers (default: as many as --gpus lists)
#   --gpus LIST        comma-separated GPU ids, one per worker, round-robin if
#                      fewer than --workers (default 0)
#   --until "TS"       stop deadline. Checked AFTER each file completes, so a
#                      file in flight is always finished, never truncated.
#                      Any `date -d` form: "2026-09-16 08:16", "tomorrow 08:16", "+6 hours".
#   --limit N          stop after N files per worker (smoke tests)
#   --redo             re-translate even if the outputs already exist
#   --min-coverage F   quarantine a result that consumed less than F of its source
#                      lines (default 0.10). The align stop is working as designed on
#                      out-of-domain files -- one measured case reached 427 of 10,847
#                      source lines at a 94% miss rate -- but such a result is near
#                      empty, and publishing it would let --skip-done lock it in
#                      forever. Quarantined ids go to .work/quarantine/ with their log
#                      and are RETRIED on the next run. Pass 0 to publish everything.
#   --ids FILE         translate ONLY the ids listed in FILE, one per line (bare id or
#                      filename; blank lines and #comments ignored). Without it every
#                      file in raw/ is work. An id with no file in raw/ is an error, not
#                      a silent skip -- a typo would otherwise read as a short run.
#   --list-only        print what would run, then exit
#
# Skip-done: a file counts as done when BOTH rubato/ and regular/ hold a non-empty
# output for it. Partial writes cannot be mistaken for finished ones -- the converter
# and translate step both write to <path>.part and rename only on success.

set -u

REPO=/home/claude/work/deep-starry
IP=/home/claude/work/intelli-piano
RUN=training/midi/20260909-midi-translator-nota1m0909-lines360-l16d256
BASE=/home/claude/datasets/midi/piano0909
RAW=$BASE/raw
RUBATO=$BASE/rubato
REGULAR=$BASE/regular
WORK=$BASE/.work
NODE_BIN=/home/claude/.nvm/versions/node/v24.10.0/bin

# Decode geometry: --src-window 960 is the measured value for this [64,512] l16d256
# run (640 is under-sized); --prime-window 320 is already right. --max-steps 0 = no cap.
# --align-advance is required by --annotate-source. The two reuse/density guards are the
# ba002e5 over-generation axes, off by default upstream, on here.
SRC_WINDOW=960
PRIME_WINDOW=320
STOP_WINDOW=16
STOP_RATE=0.6
TRIM_RATE=0.3
STOP_REUSE=0.34
# The reuse axis reads its OWN window, longer than the miss axis's 16. Bar-to-bar repetition is
# invisible inside one bar's worth of notes: measured on 005c15c173, the model repeats one bar 5
# times and every 16-note window still reads distinct/matched 0.85-1.00 while the run re-uses source
# 1.51x. 48 is where the window first spans the ~3 repetitions the 0.34 threshold needs, and over 9
# files nothing healthy produces even one sub-threshold window.
STOP_REUSE_WINDOW=48
TRIM_DENSITY=2.0
# The one trim signal reuse cannot inflate: ReuseCost is 0.0, so a bar that replays a 2-note span
# nine times books nine clean matches and reads a PERFECT miss ratio -- measured on a4956a41f6, whose
# last kept bar the trim saw as seen=9 missed=0 ratio=0.00 with no flag at all. cd83f973b2 is the same
# shape, and its whole 13-bar loop sat at miss 0.28, under TRIM_RATE, so the miss axis was blind to
# all of it. Read ONLY at the bar the backward walk would stop at, because the healthy and bad
# distributions overlap (healthy p95 2.00x, p99 4.00x, max 11.00x; the flagged bars 1.53x-9.00x) --
# there a false positive gives up one trailing bar instead of everything behind a mid-file cut.
# VERIFIED at 4.0 by re-running the affected files: a4956a41f6 9 bars -> 8 and cd83f973b2 21 -> 20,
# the two targets, plus 3 bars of collateral on 80b813fe19 (of 50) and NONE on 00a114b763. A
# published-file proxy had predicted 1 each on those two, so it is good enough to pick the threshold
# and not to predict a per-file cost: the in-tool `seen` counts every generated note in the bar while
# the proxy counts only what survived into the file, and the guard drops a RUN once it refuses to stop.
TRIM_SPAN_RATIO=4.0
# Cut at the start of the first run of this many consecutive bars with no source span -- the only
# trim axis that can act on the MIDDLE of a file. The other two walk backwards and stop at the first
# healthy bar, so a loop with healthy bars after it is invisible to them. A RUN because a lone empty
# bar is normal (the tie convention must empty one of two bars sharing a boundary): over 9 files, 12
# runs of length 1 and 4 of length 2 against exactly one of length 4, the loop.
#
# The cost is REAL and accepted here: everything after the run goes too. Measured on 005c15c173,
# 62 bars -> 48 and coverage 653/670 -> 511/670. The trade is deliberate -- a bar-aligned pair is
# the point of this corpus, and bars that are read against source a loop already mis-attributed are
# not usable data whatever their coverage says.
# 3, not the 4 it was set to when this axis landed. That value came from a single instance
# (005c15c173's 5-bar loop) and the run-length distribution over 9 files. RE-MEASURED over the 39
# published pairs: 49 runs of length 1, 4 of length 2, 2 of length 3, 1 of length 7. 3 now has a
# positive case that did not exist before -- b01a18f48a, where bars 3-5 get no source span, bar 6
# recovers and ends the backward walk, so bars 2-5 all survived -- and its only collateral is
# 00db9e936e cutting at bar 94 of 97, a tail the trim would drop anyway. 2 would additionally reach
# 6b8fc07a63's bars 21-22 but costs 005c15c173 35 of its 49 bars and 8b1d255895 9 of 39: not worth it.
TRIM_SPANLESS_RUN=3

WORKERS=""
GPUS="0"
UNTIL=""
LIMIT=0
REDO=0
LIST_ONLY=0
MIN_COVERAGE=0.10
IDS_FILE=""

while [ $# -gt 0 ]; do
	case "$1" in
		--workers) WORKERS="$2"; shift 2 ;;
		--gpus) GPUS="$2"; shift 2 ;;
		--until) UNTIL="$2"; shift 2 ;;
		--limit) LIMIT="$2"; shift 2 ;;
		--redo) REDO=1; shift ;;
		--min-coverage) MIN_COVERAGE="$2"; shift 2 ;;
		--ids) IDS_FILE="$2"; shift 2 ;;
		--list-only) LIST_ONLY=1; shift ;;
		-h|--help) sed -n '2,30p' "$0"; exit 0 ;;
		*) echo "unknown option: $1" >&2; exit 2 ;;
	esac
done

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
[ -z "$WORKERS" ] && WORKERS=${#GPU_ARR[@]}

# Resolve the deadline ONCE, up front: a bad --until must fail before any GPU work,
# not silently never trigger.
DEADLINE=0
if [ -n "$UNTIL" ]; then
	if ! DEADLINE=$(date -d "$UNTIL" +%s 2>/dev/null); then
		echo "cannot parse --until '$UNTIL'" >&2; exit 2
	fi
	NOW=$(date +%s)
	if [ "$DEADLINE" -le "$NOW" ]; then
		echo "--until '$UNTIL' ($(date -d "@$DEADLINE" '+%F %T')) is already past" >&2; exit 2
	fi
	echo "deadline: $(date -d "@$DEADLINE" '+%F %T') ($(( (DEADLINE-NOW)/3600 ))h $(( ((DEADLINE-NOW)%3600)/60 ))m from now)"
fi

QUAR="$WORK/quarantine"
mkdir -p "$RUBATO" "$REGULAR" "$WORK" "$QUAR"

# ---- work list ----------------------------------------------------------------
ALL=$(ls -1 "$RAW" 2>/dev/null | grep -E '\.midi?$' | sort)

# --ids restricts the work to a named set. Resolved against raw/ HERE, before any GPU is
# touched, and a name with no file is fatal: a typo that silently dropped an id would come
# back as a short run that looks like it succeeded.
if [ -n "$IDS_FILE" ]; then
	[ -r "$IDS_FILE" ] || { echo "--ids: cannot read '$IDS_FILE'" >&2; exit 2; }
	WANT=$(sed -e 's/#.*//' -e 's/[[:space:]]//g' "$IDS_FILE" | grep -v '^$' | sed 's/\.midi\?$//' | sort -u)
	[ -n "$WANT" ] || { echo "--ids: '$IDS_FILE' lists no ids" >&2; exit 2; }
	SEL="" MISSING=""
	for id in $WANT; do
		f=$(printf '%s\n' "$ALL" | grep -m1 -E "^${id}\.midi?$" || true)
		if [ -n "$f" ]; then SEL="$SEL$f"$'\n'; else MISSING="$MISSING $id"; fi
	done
	if [ -n "$MISSING" ]; then
		echo "--ids: no file in raw/ for:$MISSING" >&2; exit 2
	fi
	ALL=$(printf '%s' "$SEL" | grep . | sort)
	echo "--ids $IDS_FILE: $(printf '%s\n' "$ALL" | grep -c .) of $(printf '%s\n' "$WANT" | grep -c .) requested resolved"
fi

TOTAL=$(printf '%s\n' "$ALL" | grep -c . )

TODO=""
for f in $ALL; do
	id=${f%.*}
	if [ "$REDO" -eq 0 ] && [ -s "$RUBATO/$id.midiseq2.txt" ] && [ -s "$REGULAR/$id.midiseq2.txt" ]; then
		continue
	fi
	TODO="$TODO$f"$'\n'
done
NTODO=$(printf '%s' "$TODO" | grep -c . )

echo "raw $TOTAL files, $NTODO to do, $((TOTAL-NTODO)) already done"
echo "workers $WORKERS, gpus ${GPU_ARR[*]}"

if [ "$LIST_ONLY" -eq 1 ]; then
	printf '%s' "$TODO" | head -20
	[ "$NTODO" -gt 20 ] && echo "... and $((NTODO-20)) more"
	exit 0
fi
[ "$NTODO" -eq 0 ] && { echo "nothing to do"; exit 0; }

# Deal the work list out round-robin so every worker gets a mix of file sizes;
# a contiguous split would hand one worker all the big files.
printf '%s' "$TODO" > "$WORK/todo.all"
rm -f "$WORK"/todo.w*
i=0
while IFS= read -r f; do
	[ -z "$f" ] && continue
	echo "$f" >> "$WORK/todo.w$((i % WORKERS))"
	i=$((i+1))
done < "$WORK/todo.all"

# ---- one worker ---------------------------------------------------------------
worker () {
	local WID=$1 GPU=$2 LIST=$3
	local done_n=0 skip_n=0 fail_n=0 quar_n=0
	local CONV_IN="$WORK/conv_in.$WID" CONV_OUT="$WORK/conv_out.$WID"

	# Persistent converter: a fifo pair so one ts-node process serves every file
	# this worker touches. Boot cost is paid once here, not 4,000 times.
	rm -f "$CONV_IN" "$CONV_OUT"
	mkfifo "$CONV_IN" "$CONV_OUT"
	(
		cd "$IP" || exit 1
		export PATH=$NODE_BIN:$PATH
		npx ts-node --project ./tsconfig.node.json midiToSeq2Server.ts < "$CONV_IN" > "$CONV_OUT" 2>"$WORK/conv_err.$WID"
	) &
	local CONV_PID=$!
	# Hold the write end open for the whole worker, else the server sees EOF after file 1.
	exec {CONV_FD}> "$CONV_IN"
	exec {CONV_RD}< "$CONV_OUT"

	cleanup () {
		exec {CONV_FD}>&- 2>/dev/null
		exec {CONV_RD}<&- 2>/dev/null
		kill "$CONV_PID" 2>/dev/null
		rm -f "$CONV_IN" "$CONV_OUT"
	}
	trap cleanup EXIT

	while IFS= read -r f; do
		[ -z "$f" ] && continue
		local id=${f%.*}
		local src="$WORK/$id.midiseq2.txt"

		if [ "$REDO" -eq 0 ] && [ -s "$RUBATO/$id.midiseq2.txt" ] && [ -s "$REGULAR/$id.midiseq2.txt" ]; then
			skip_n=$((skip_n+1)); continue
		fi

		local t0=$(date +%s)

		# --- midi -> midiseq2, on the spot ---
		printf '%s\t%s\n' "$RAW/$f" "$src" >&$CONV_FD
		local reply=""
		IFS= read -r reply <&$CONV_RD
		case "$reply" in
			OK*) : ;;
			*) echo "[w$WID] CONVFAIL $id ${reply#ERR }"; fail_n=$((fail_n+1)); continue ;;
		esac
		if [ ! -s "$src" ]; then
			echo "[w$WID] CONVEMPTY $id"; fail_n=$((fail_n+1)); rm -f "$src"; continue
		fi

		# --- translate ---
		# Bare `python3` off a PATH-first venv so ps/nvidia-smi show short forms, and a
		# RELATIVE run dir via the `training` symlink, for the same reason.
		( cd "$REPO" && \
		  PATH=$REPO/venv/bin:$PATH PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU \
		  python3 tools/midi/translateMidiseq2.py \
			--run "$RUN" \
			--input "$src" \
			--output "$REGULAR/$id.midiseq2.txt.part" \
			--annotate-source "$WORK/ann.$WID" \
			--src-window $SRC_WINDOW --prime-window $PRIME_WINDOW --max-steps 0 \
			--align-advance \
			--align-stop-window $STOP_WINDOW --align-stop-rate $STOP_RATE \
			--align-trim-rate $TRIM_RATE \
			--align-stop-reuse $STOP_REUSE --align-stop-reuse-window $STOP_REUSE_WINDOW \
			--align-trim-density $TRIM_DENSITY \
			--align-trim-spanless-run $TRIM_SPANLESS_RUN \
			--align-trim-span-ratio $TRIM_SPAN_RATIO \
			--device cuda ) > "$WORK/log.$WID.$id" 2>&1
		local rc=$?

		local ann="$WORK/ann.$WID/$(basename "$src")"

		# Coverage gate: the tool prints "[warn] stopped after <used>/<total> source lines"
		# when the stop fired early. Read it rather than guessing from output length --
		# a legitimately short piece is not the same as a collapsed run.
		# The honest coverage, appended for EVERY file that produced a log: `stopped after`
		# is the cursor (measured 3.3-4x inflated), `furthest N/M` from [align-advance] is
		# the distinct source notes really reached. The gate below still reads the cursor --
		# the threshold has not been settled -- so this manifest is what a later filter uses,
		# and it must be written before the success path deletes the log.
		if [ -f "$WORK/log.$WID.$id" ]; then
			local fw ft
			fw=$(sed -n 's/.*furthest \([0-9]*\)\/\([0-9]*\).*/\1/p' "$WORK/log.$WID.$id" | tail -1)
			ft=$(sed -n 's/.*furthest \([0-9]*\)\/\([0-9]*\).*/\2/p' "$WORK/log.$WID.$id" | tail -1)
			if [ -n "$fw" ] && [ -n "$ft" ] && [ "$ft" -gt 0 ]; then
				printf '%s\t%s\t%s\t%s\n' "$id" "$fw" "$ft" \
					"$(awk -v u="$fw" -v t="$ft" 'BEGIN{printf "%.4f", u/t}')" \
					>> "$WORK/coverage.tsv"
			fi
		fi

		local cover_ok=1 cover_txt=""
		if [ "$MIN_COVERAGE" != "0" ] && [ -f "$WORK/log.$WID.$id" ]; then
			local used total
			used=$(sed -n 's/.*stopped after \([0-9]*\)\/\([0-9]*\) source lines.*/\1/p' "$WORK/log.$WID.$id" | tail -1)
			total=$(sed -n 's/.*stopped after \([0-9]*\)\/\([0-9]*\) source lines.*/\2/p' "$WORK/log.$WID.$id" | tail -1)
			if [ -n "$used" ] && [ -n "$total" ] && [ "$total" -gt 0 ]; then
				if [ "$(awk -v u="$used" -v t="$total" -v m="$MIN_COVERAGE" 'BEGIN{print (u/t < m) ? 1 : 0}')" = "1" ]; then
					cover_ok=0
					cover_txt="$used/$total"
				fi
			fi
		fi

		if [ $rc -eq 0 ] && [ -s "$REGULAR/$id.midiseq2.txt.part" ] && [ -s "$ann" ] && [ "$cover_ok" -eq 0 ]; then
			# Reached the model's limit, not an error. Keep the evidence, publish nothing,
			# so the next run retries instead of trusting a near-empty result.
			mkdir -p "$QUAR"
			mv -f "$WORK/log.$WID.$id" "$QUAR/$id.log" 2>/dev/null
			mv -f "$REGULAR/$id.midiseq2.txt.part" "$QUAR/$id.regular.midiseq2.txt" 2>/dev/null
			mv -f "$ann" "$QUAR/$id.rubato.midiseq2.txt" 2>/dev/null
			quar_n=$((quar_n+1))
			echo "[w$WID gpu$GPU] LOWCOVER $id $cover_txt -- quarantined, not published"
			rm -f "$src"
			if [ "$DEADLINE" -ne 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
				echo "[w$WID gpu$GPU] deadline reached, stopping"; break
			fi
			continue
		fi

		if [ $rc -eq 0 ] && [ -s "$REGULAR/$id.midiseq2.txt.part" ] && [ -s "$ann" ]; then
			# Publish both arms only once both exist, so a killed worker never leaves
			# a regular/ file with no rubato/ counterpart for --skip-done to trust.
			mv -f "$REGULAR/$id.midiseq2.txt.part" "$REGULAR/$id.midiseq2.txt"
			mv -f "$ann" "$RUBATO/$id.midiseq2.txt"
			done_n=$((done_n+1))
			local dt=$(( $(date +%s) - t0 ))
			echo "[w$WID gpu$GPU] ok $id ${dt}s ($(grep -c . "$REGULAR/$id.midiseq2.txt") lines)"
			# KEPT, not deleted: the log is the only record of HOW the run ended -- `done`
			# (natural <eos>) vs the align stop vs the tail trim vs an exhausted window.
			# The tool prints [align-stop]/[align-trim]/[warn] stopped after only when it did
			# NOT finish, so a log with none of them is the only positive evidence of a
			# natural completion. Deleting it made that question unanswerable after the fact.
			mkdir -p "$WORK/logs" && mv -f "$WORK/log.$WID.$id" "$WORK/logs/$id.log"
		else
			fail_n=$((fail_n+1))
			echo "[w$WID gpu$GPU] FAIL $id rc=$rc -- see $WORK/log.$WID.$id"
			rm -f "$REGULAR/$id.midiseq2.txt.part"
		fi
		rm -f "$src"

		# --- deadline, checked after the file is safely published ---
		if [ "$DEADLINE" -ne 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
			echo "[w$WID gpu$GPU] deadline reached, stopping"
			break
		fi
		if [ "$LIMIT" -ne 0 ] && [ "$done_n" -ge "$LIMIT" ]; then
			echo "[w$WID gpu$GPU] limit $LIMIT reached, stopping"
			break
		fi
	done < "$LIST"

	cleanup
	trap - EXIT
	echo "[w$WID gpu$GPU] done=$done_n skipped=$skip_n failed=$fail_n quarantined=$quar_n"
}

# ---- launch ------------------------------------------------------------------
PIDS=""
for w in $(seq 0 $((WORKERS-1))); do
	LIST="$WORK/todo.w$w"
	[ -s "$LIST" ] || continue
	GPU=${GPU_ARR[$((w % ${#GPU_ARR[@]}))]}
	mkdir -p "$WORK/ann.$w"
	worker "$w" "$GPU" "$LIST" &
	PIDS="$PIDS $!"
done

for p in $PIDS; do wait "$p"; done

echo "=== all workers finished ==="
echo "rubato : $(ls -1 "$RUBATO" | grep -c 'midiseq2.txt$') files"
echo "regular: $(ls -1 "$REGULAR" | grep -c 'midiseq2.txt$') files"
QN=$(ls -1 "$QUAR" 2>/dev/null | grep -c '\.log$')
[ "$QN" -gt 0 ] && echo "quarantined (low coverage, will retry): $QN -- see $QUAR"
