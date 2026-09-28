#!/bin/bash
cd /home/vilem/Projects/brats-upenn-gbm
LOG=/home/vilem/.claude/jobs/63a02e34/tmp/cohort.log
: > "$LOG"
for N in 52 84 88 93 122 130 133 140 141 145 148 150 160 183 197 219 267 280 295 301 302 312 314 320 332 344 354 355; do
  for T in 1 2; do
    printf '=== %s tp%s %s\n' "$N" "$T" "$(date +%H:%M)" >> "$LOG"
    ./main.py "$N" --timepoint "$T" --no-view >> "$LOG" 2>&1 \
      || printf '!!! %s tp%s FAILED\n' "$N" "$T" >> "$LOG"
  done
done
printf 'ALL DONE %s\n' "$(date +%H:%M)" >> "$LOG"
