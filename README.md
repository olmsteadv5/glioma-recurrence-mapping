# BraTS segmentation on UPenn-GBM

Runs the [BrainLesion/BraTS](https://github.com/BrainLesion/BraTS) competition
algorithms over subjects from the UPenn-GBM cohort and scores the results
against the cohort's expert reference segmentations where they exist.

## The GPU caveat, up front

**The 1st-place BraTS 2025 algorithm cannot run on this machine.** This laptop
has no NVIDIA GPU, and upstream ships `BraTS25_1` as CUDA-only.

| Algorithm | Rank | Runs on CPU? | Upstream CPU estimate |
|---|---|---|---|
| `BraTS25_1` (Jain et al.) | 1st | no — CUDA only | — |
| `BraTS25_2` (Qu Lin et al.) | 2nd | yes | ~7 min/subject |
| `BraTS25_3A` (Liwei Jin et al.) | 3rd (tie) | yes | ~150 min/subject |
| `BraTS25_3B` (Celaya et al., MIST) | 3rd (tie) | no — CUDA only | — |

`run_segmentation.py` therefore **defaults to `BraTS25_1` and falls back to
`BraTS25_2`**, printing the substitution loudly. `BraTS25_2` is the same
algorithm Ryan's original script used — not a coincidence, it is the best
entry that runs without a GPU.

To actually get the winning model, run this project on a CUDA box (a Yale
cluster node, or any cloud GPU instance). Nothing else changes:

```bash
python scripts/run_segmentation.py            # picks BraTS25_1 automatically
python scripts/run_segmentation.py --require-algorithm   # refuse to fall back
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install brats nibabel numpy
```

Docker must be running and your user must be in the `docker` group. The
algorithm images are large (multi-GB) and pull on first use.

## Usage

```bash
# 1. fetch subjects (defaults to 36 42 51 — the ones with existing tractography)
.venv/bin/python scripts/fetch_data.py 36 42 51 --with-seg
.venv/bin/python scripts/fetch_data.py 2 --with-seg     # has a reference seg

# 2. sanity-check the inputs before burning CPU hours
.venv/bin/python scripts/verify_inputs.py

# 3. segment
.venv/bin/python scripts/run_segmentation.py --dry-run
.venv/bin/python scripts/run_segmentation.py

# 4. volumes + Dice against reference labels
.venv/bin/python scripts/evaluate.py
```

### Why `--jobs 1`

Ryan's original script used `ProcessPoolExecutor(max_workers=4)`, i.e. four
nnU-Net-class containers at once. That is right on a large GPU server and
wrong here: this machine has 22 GB RAM with ~6 GB free, so four CPU containers
would swap and finish slower than one at a time. `--jobs` defaults to 1 and
should only be raised where there is real headroom.

## Layout

```
data/PENN_GBM_nii/<SUBJECT>/<SUBJECT>_{T1,T1GD,T2,FLAIR}.nii.gz   inputs
data/reference_seg/<SUBJECT>_seg.nii.gz                           expert labels
data/manifest.json                    provenance + sha256 per downloaded file
outputs/<algorithm>/<SUBJECT>_seg.nii.gz
outputs/<algorithm>/run_summary.json
outputs/evaluation.json
logs/<algorithm>/<SUBJECT>.log        per-subject container logs
```

The `data/PENN_GBM_nii/<SUBJECT>/<SUBJECT>_<MODALITY>.nii.gz` layout matches
the convention in Ryan's original script, so it stays drop-in compatible.

## Modality mapping

BraTS names modalities differently from UPenn-GBM. The runner maps:

| BraTS arg | Meaning | UPenn-GBM file |
|---|---|---|
| `t1c` | post-contrast T1 | `_T1GD.nii.gz` |
| `t1n` | native T1 | `_T1.nii.gz` |
| `t2f` | T2-FLAIR | `_FLAIR.nii.gz` |
| `t2w` | T2 | `_T2.nii.gz` |

Getting `t1c`/`t1n` backwards is the easy mistake here and does not error —
it just quietly produces a bad enhancing-tumour label.

## Labels

BraTS 2024/2025 adult glioma:

| Value | Region |
|---|---|
| 1 | NETC — non-enhancing tumour core (necrosis) |
| 2 | SNFH — surrounding non-enhancing FLAIR hyperintensity (oedema) |
| 3 | ET — enhancing tissue |
| 4 | RC — resection cavity (post-treatment only) |

Scored as the standard nested regions: `ET = {3}`, `TC = {1,3}`,
`WT = {1,2,3}`.

## Data provenance — read before publishing

Inputs come from a **public GitHub-release mirror** of TCIA
(`data-nih/tcia`, tag `upenn-gbm`), because the official TCIA route requires
the NBIA Data Retriever / Aspera client and does not script cleanly.

Verified properties of the mirrored files: 240×240×155, 1 mm isotropic, LPS
(SRI24 atlas space), skull-stripped — i.e. already in BraTS preprocessed form,
no further preprocessing needed.

Two caveats:

1. **This is not an official TCIA endpoint.** It is an unaffiliated account
   with no institutional backing. Fine for a pilot; re-pull from TCIA before
   anything leaves the lab.
2. **The mirrored volumes are 8-bit (uint8, 0–255).** Whether that matches the
   TCIA originals or is a quantisation introduced by the mirror has not been
   confirmed. BraTS models z-score normalise per modality, so this is unlikely
   to change results much, but it should be checked against a TCIA original
   before any published number rests on it.

Subject id mapping, verified against `participants.tsv`:
`sub-NNN` ↔ `UPENN-GBM-00NNN_11` (baseline), `sub-NNN_2` ↔ `..._21` (later scan).

Cite: Bakas et al., *The University of Pennsylvania glioblastoma (UPenn-GBM)
cohort*, Scientific Data 9, 453 (2022).

## First run — results (2026-09-14)

`BraTS25_2` on CPU, `--jobs 1`. Four subjects, **~4.5 min each** (faster than the
~7 min upstream estimate).

| Subject | WT mL | TC mL | ET mL | Dice WT | Dice TC | Dice ET |
|---|---|---|---|---|---|---|
| UPENN-GBM-00002_11 | 308.8 | 53.4 | 37.7 | 0.757 | 0.882 | 0.812 |
| UPENN-GBM-00036_11 | 226.9 | 42.8 | 22.1 | — | — | — |
| UPENN-GBM-00042_11 | 155.6 | 73.9 | 43.7 | — | — | — |
| UPENN-GBM-00051_11 |  43.8 | 16.2 |  9.2 | — | — | — |

Only subject 2 has published reference labels; the other three are the subjects
with existing tractography (`~/penn_gbm_tractography/fib_files/`).

### Reading the subject-2 Dice

`TC 0.882` / `ET 0.812` land in the published BraTS-winner range, which is the
result that validates the pipeline — in particular it confirms the `t1c`/`t1n`
mapping is right, since swapping those craters ET.

`WT 0.757` is lower, and the decomposition explains it:

```
WT recall    0.930   finds 93% of the reference tumour
WT precision 0.636   but 36% of what it calls tumour isn't in the reference

per-label volume ratio (pred/ref):  NETC 1.02   SNFH 1.52   ET 1.36
```

The model is not missing tumour, it is adding. Necrotic core matches almost
exactly (ratio 1.02); the excess is nearly all SNFH. The QC overlay shows the
model's oedema extending further than the reference while still tracking
visible FLAIR hyperintensity.

That pattern points at an annotation-protocol difference rather than a model
failure: the UPenn-GBM labels are expert-*approved* semi-automated output, and
the oedema boundary is the softest call in the label set. Dice against a
different protocol's labels is not Dice on the BraTS test set. The 8-bit
quantisation noted above could also shift a soft intensity threshold — testable
against a TCIA original if a result depends on it.

Practical reading: **trust TC/ET; treat WT as protocol-dependent.** A consistent
oedema bias cancels in longitudinal volume tracking, but would matter for
radiotherapy target definition.

### Visual QC

All four overlays (`outputs/BraTS25_2/qc/`) are anatomically sound: enhancing
label on the enhancing rim, necrosis filling the core, oedema on FLAIR-bright
tissue. Subjects 36 and 51 are textbook ring-enhancing lesions; 42 is a bulky
frontal mass with interdigitating enhancing septa, traced correctly.

## Viewing results in ITK-SNAP

```bash
./view.sh UPENN-GBM-00036_11              # prediction over T1GD
./view.sh UPENN-GBM-00002_11 reference    # the expert labels instead
```

This loads T1GD as the main image, the segmentation as a labelled overlay with
proper region names and colours (`brats_labels.itksnap.txt`), and T1/T2/FLAIR
as additional layers.

Once open:

- The **modality dropdown** (top-left) cycles T1GD / T1 / T2 / FLAIR with the
  segmentation staying put. This is the check that matters — flip to T1GD and
  the ET label should sit on bright rim; flip to FLAIR and SNFH should sit on
  bright tissue.
- **`a`** toggles the segmentation overlay on and off.
- **`s`**, **`c`**, **`a`** panes are sagittal / coronal / axial; scroll to move
  through slices.
- The **3D pane** (bottom right) renders the tumour surface after you press
  *Update* in the segmentation panel.

Equivalent raw command, if you prefer not to use the script:

```bash
itksnap -g data/PENN_GBM_nii/SUBJ/SUBJ_T1GD.nii.gz \
        -s outputs/BraTS25_2/SUBJ_seg.nii.gz \
        -l brats_labels.itksnap.txt
```
