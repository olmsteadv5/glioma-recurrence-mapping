# Brain tumor segmentation on UPenn-GBM

Segments glioblastoma MRI from UPenn-GBM, maps each tumor onto the Yale Brain Atlas and HCP-1065 tracts, and measures how much of a later recurrence the original tumor and its margins cover.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

You also need Docker (your user in the docker group), FreeSurfer at `/opt/freesurfer` for SynthStrip and SynthSeg, and `greedy` on your PATH.

The HCP-1065 tract atlas is not downloaded for you. Unzip [the tracts](https://github.com/data-others/atlas/releases/download/hcp1065/hcp1065_avg_tracts_trk.zip) into `inputs/atlas/hcp1065/trk/`, and put [ICBM152_adult.T1W.nii.gz](https://github.com/data-others/atlas/releases/download/ICBM152_adult/ICBM152_adult.T1W.nii.gz) and [ICBM152_adult.mask.nii.gz](https://github.com/data-others/atlas/releases/download/ICBM152_adult/ICBM152_adult.mask.nii.gz) in `inputs/atlas/hcp1065/`.

## Usage

```bash
./main.py 55                  # pre-op scan of patient 55
./main.py 55 --timepoint 2    # its follow-up
./main.py 55 --from segment   # rerun from one stage
./tui.py                      # browse results, needs the kitty terminal
```

Stages run in order: fetch, strip, tissue, segment, register, align, map. Segmentation uses the BraTS 2025 winner on an NVIDIA GPU and the second-place model on CPU. The first run pulls a 30 GB Docker image.

The scripts in `analysis/` make the tables and figures from processed pairs, into `outputs/results/native/` and `outputs/results/mni/`. Start with `overlap_csv.py`, `capture_curves.py` and `walkdown.py`.
