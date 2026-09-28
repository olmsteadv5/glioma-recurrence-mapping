#!/bin/bash
# after 00042 and 00051: tissue maps for the second scans processed before the
# tissue step existed, then the full audit again. One SynthSeg at a time.
cd /home/vilem/Projects/brats-upenn-gbm
T=/home/vilem/.claude/jobs/63a02e34/tmp
while pgrep -f cohort3.sh >/dev/null; do sleep 20; done
for n in 00036 00045 00055 00086 00128 00129 00134 00285 00307 00352 00353; do
  printf '=== %s_21 tissue %s\n' "$n" "$(date +%H:%M)" >> $T/finish.log
  .venv/bin/python -c "from scan import tissue; tissue.to_mni('UPENN-GBM-${n}_21')" >> $T/finish.log 2>&1 \
    || printf '!!! %s_21 tissue FAILED\n' "$n" >> $T/finish.log
done
printf '=== audit %s\n' "$(date +%H:%M)" >> $T/finish.log
.venv/bin/python $T/_audit.py >> $T/finish.log 2>&1
printf 'ALL DONE %s\n' "$(date +%H:%M)" >> $T/finish.log
