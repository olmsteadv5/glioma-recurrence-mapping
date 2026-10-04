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
segment, register, align and map in that order. Align only does anything for a
follow-up: it lines the same patient's pre-op scan up with it, rigidly, so
the two can be compared without leaving the patient's own space. Use `--from segment` (or any stage) to
redo a scan from that point, and `--no-view` to skip the browser window.

Each scan takes about 5 minutes on CPU, mostly segmentation. The first run is
slower because it downloads a 30 GB Docker image. Segmentation uses the BraTS
2025 winning model on an NVIDIA GPU and the second-place model on CPU.

The 82 segmentations in `outputs/` are from the winning model (BraTS25_1), run
on a Yale Bouchet GPU because this laptop has none; each scan's
`segmentation.log` says how. The earlier second-place (CPU) segmentations and
the results made from them are kept in `outputs_25_2/`.

## The overlap results

Every result is made twice, into `outputs/native/` and `outputs/mni/`. Native is
the patient's own follow-up scan, where volumes are exactly as scanned; MNI is the
standard brain the atlas lives in, which stretches volumes a little.

```bash
./analysis/overlap_csv.py        # overlap_review.csv, one row per patient
./analysis/capture_curves.py     # how much a uniform margin catches, out to the farthest recurrence
./analysis/pathology_figures.py  # where the tumor outside 20 mm goes, and what the 2 cm margin spends
./analysis/overlap_plots.py      # overlap against tumor size, and margins by pathology score
./analysis/lobes.py              # tumor_lobes.csv, the lobes each tumor sits in
./analysis/walkdown.py           # a target 20 mm along the tracts leaving the tumor, against 2 cm and a same-size ball
./analysis/walkdown_figures.py   # the two figures for sending: cost and capture of each target, and walk-down vs ball
```

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
