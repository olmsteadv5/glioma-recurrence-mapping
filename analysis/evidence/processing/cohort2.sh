#!/bin/bash
# resume after the crash: each subject starts at the stage it died in. Nothing
# else runs alongside, since the first pass lost ten SynthSeg runs to the OOM killer
cd /home/vilem/Projects/brats-upenn-gbm
LOG=/home/vilem/.claude/jobs/63a02e34/tmp/cohort2.log
: > "$LOG"
run() {
  printf '=== %s tp%s from %s %s\n' "$1" "$2" "$3" "$(date +%H:%M)" >> "$LOG"
  ./main.py "$1" --timepoint "$2" --from "$3" --no-view >> "$LOG" 2>&1 \
    || printf '!!! %s tp%s FAILED\n' "$1" "$2" >> "$LOG"
}
# failed at tissue: fetch and strip are on disk
for pair in "52 1" "122 2" "130 2" "141 1" "141 2" "148 1" "148 2" "183 1" "197 1" "219 1"; do
  run $pair tissue
done
# interrupted mid-segmentation when the machine went down
run 219 2 segment
# never started
for N in 267 280 295 301 302 312 314 320 332 344 354 355; do
  run "$N" 1 fetch
  run "$N" 2 fetch
done
printf 'ALL DONE %s\n' "$(date +%H:%M)" >> "$LOG"
