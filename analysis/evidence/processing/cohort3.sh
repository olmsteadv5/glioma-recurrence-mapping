#!/bin/bash
# the two pairs the first to-do list missed: their first scans predate the
# tissue step, and their second scans were never fetched
cd /home/vilem/Projects/brats-upenn-gbm
LOG=/home/vilem/.claude/jobs/63a02e34/tmp/cohort3.log
: > "$LOG"
run() {
  printf '=== %s tp%s from %s %s\n' "$1" "$2" "$3" "$(date +%H:%M)" >> "$LOG"
  ./main.py "$1" --timepoint "$2" --from "$3" --no-view >> "$LOG" 2>&1 \
    || printf '!!! %s tp%s FAILED\n' "$1" "$2" >> "$LOG"
}
run 42 1 tissue
run 42 2 fetch
run 51 1 tissue
run 51 2 fetch
printf 'ALL DONE %s\n' "$(date +%H:%M)" >> "$LOG"
