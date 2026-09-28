# Brain tumor segmentation on UPenn-GBM

This takes glioblastoma MRI from the UPenn-GBM collection and processes each
scan: skull strip, tissue labels, tumor segmentation, and registration onto a
standard brain. From there it works out which Yale Brain Atlas regions the tumor
sits in and which white matter tracts run through it.

41 patients have a scan before surgery and a later follow-up. For those, it
measures how much of the later tumor overlaps the original one.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

You also need Docker (and to be in the docker group), FreeSurfer for SynthStrip
and SynthSeg, and greedy, which comes with ITK-SNAP.

## Processing a scan

```bash
./main.py 55                 # the pre-surgery scan
./main.py 55 --timepoint 2   # the follow-up
```

The number is the UPenn-GBM patient number. It runs fetch, strip, tissue,
segment, register and map in that order. Use `--from segment` (or any stage) to
redo a scan from that point, and `--no-view` to skip the browser window.

Each scan takes about 5 minutes on CPU, mostly segmentation. The first run is
slower because it downloads a 30 GB Docker image. Segmentation uses the BraTS
2025 winning model on an NVIDIA GPU and the second-place model on CPU.

## The overlap results

```bash
./analysis/overlap_csv.py
```

This writes `outputs/overlap_review.csv`, one row per patient.
`analysis/COHORT.md` explains how the 41 patients were chosen and checked.

## Looking at a patient

```bash
./tui.py                  # terminal app, needs the kitty terminal
python -m show.view 55    # 3D view in the browser
```

`show/README.md` has the details.

## What's where

```
scan/       processing each scan
yale/       the Yale Brain Atlas
analysis/   the overlap results
show/       the viewers
inputs/     raw downloads      (not in git)
outputs/    everything made from them (not in git)
```

Everything in `inputs/` and `outputs/` can be regenerated.
