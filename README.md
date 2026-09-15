# BraTS segmentation on UPenn-GBM, mapped onto the Yale Brain Atlas

Segments glioma MRI with the [BraTS](https://github.com/BrainLesion/BraTS)
competition algorithms, then reports which Yale Brain Atlas parcels each tumour
occupies.

```bash
python main.py 36 42 51
```

Fetches the scans, skull-strips, verifies, segments, scores against reference
labels where they exist, registers into MNI, intersects with the atlas, and
opens a 3D view per subject. Every stage runs each time; only the downloads and
the cached atlas parcellation are reused.

To re-open a view without recomputing, from a PyCharm console:

```python
import json, atlas, view
yba, viz = atlas.load_atlas()
results = json.load(open("outputs/results.json"))
view.run("UPENN-GBM-00036_11", yba, viz, results["UPENN-GBM-00036_11"]["atlas"])
```

`view` follows the project's `add`/`run` contract, so `view.add(subject, viz)`
layers a tumour onto a figure another module already started.

## Setup

One interpreter runs everything:

```
/home/vilem/Projects/YBAParcelVisualizer/.venv/bin/python
```

It carries `YaleBrainAtlas`, `brats`, nibabel, pyvista, trimesh and plotly.
Point PyCharm's project interpreter at it. Docker must be running and your user
must be in the `docker` group; the algorithm image is ~30 GB and pulls on first
use.

## Layout

```
main.py        entry point
fetch.py       download from the TCIA mirror
skullstrip.py  SynthStrip
verify.py      preprocessing checks
segment.py     BraTS via Docker
evaluate.py    volumes and Dice
register.py    SRI24 -> MNI152
atlas.py       parcel voxelisation and volume overlap
view.py        3D view
paths.py       shared paths

inputs/<SUBJECT>/<SUBJECT>_{T1,T1GD,T2,FLAIR}.nii.gz   raw, never modified
inputs/reference_seg/<SUBJECT>_seg.nii.gz

outputs/<SUBJECT>/stripped/                            skull-stripped
outputs/<SUBJECT>/seg.nii.gz                           native SRI24
outputs/<SUBJECT>/seg_in_mni.nii.gz                    MNI152 / YBA space
outputs/atlas/yba_labels_mni.nii.gz                    cached parcellation
outputs/results.json                                   volumes, Dice, overlap
```

## Results

`BraTS25_2` on CPU, ~3 min per subject.

| Subject | WT mL | TC mL | ET mL | Dice WT | Dice TC | Dice ET |
|---|---|---|---|---|---|---|
| UPENN-GBM-00002_11 | 191.1 | 44.5 | 29.6 | 0.919 | 0.958 | 0.900 |
| UPENN-GBM-00036_11 | 166.6 | 40.9 | 20.8 | — | — | — |
| UPENN-GBM-00042_11 | 103.1 | 72.7 | 43.8 | — | — | — |
| UPENN-GBM-00051_11 |  35.5 | 14.7 |  7.8 | — | — | — |

Only subject 2 has published reference labels.

---

# Why the pipeline looks like this

## Skull-stripping is mandatory

The mirror ships UPenn-GBM's **unstripped** variant. BraTS algorithms assume
skull-stripped input and degrade quietly without it — plausible output, quietly
wrong. Same algorithm, same subjects, stripping the only difference:

| Dice | unstripped | stripped |
|---|---|---|
| WT | 0.757 | **0.919** |
| TC | 0.882 | **0.958** |
| ET | 0.812 | **0.900** |

Predicted whole-tumour volume went 308.8 → 191.1 mL against a 211.8 mL
reference, and spurious resection-cavity labels disappeared from pre-operative
scans.

`verify.py` checks **absolute brain volume** against a 2000 mL ceiling. A
fraction-of-grid test cannot catch this: an unstripped head is ~33% of the
240×240×155 grid, inside any plausible band for a stripped brain.

## Modality naming

BraTS names modalities differently from UPenn-GBM:

| BraTS | meaning | UPenn-GBM |
|---|---|---|
| `t1c` | post-contrast T1 | `_T1GD` |
| `t1n` | native T1 | `_T1` |
| `t2f` | T2-FLAIR | `_FLAIR` |
| `t2w` | T2 | `_T2` |

Swapping `t1c`/`t1n` does not error — it quietly ruins the enhancing-tumour
label. ET Dice is the canary.

## Algorithm choice

`BraTS25_1` (1st place, Jain et al., nnU-Net with GAN-based on-the-fly
augmentation) is CUDA-only. `segment.py` uses it when a GPU is present and falls
back to `BraTS25_2` (2nd place, the best CPU-capable entry) otherwise, saying so.
On a GPU host the winner runs with no code change.

## Registration into MNI

The YBA is defined in MNI152 — its surface spans x ±72.6, y −107.6..73.7,
z −52.4..82.5, centroid (0, −17, 16), matching the MNI152 1 mm template. BraTS
output is on the SRI24 grid, so the tumour has to move.

A glioma displaces tissue. A similarity-driven registration will squash the
lesion to make the patient resemble the template, corrupting the warp exactly
where it matters. `register.py` excludes the lesion from the deformable metric
(`-mm`, cost-function masking, standard since Brett et al. 2001).

**Two greedy quirks, both diagnosed here:**

- `-mm` **crashes in affine mode** (`Requested region is outside the largest
  possible region`). The affine stage is therefore unmasked — a 12-parameter
  global transform cannot distort locally around a lesion, so it loses little.
- `-mm` **only works with `-m NCC`**, not NMI. The flag's help text refers to a
  WNCC metric that `-m` does not accept, so WNCC is engaged internally when a
  mask accompanies NCC. Masking with NMI fails.

An earlier version worked around the crash with two passes and a template-space
`-gm` mask. The single-pass `-mm` version squashes the lesion measurably less —
Jacobian ratio of lesion to healthy tissue 1.365 vs 1.807, and tumour volume
151.6 mL vs 119.4 mL against 166.6 mL native.

Cost-function masking stops the lesion corrupting the warp; it does not undo
mass effect. Parcel assignments at the tumour rim stay approximate.

**Measure volumes in native space, not MNI.** Warping into a healthy template
squeezes out mass effect.

## What the parcel overlap measures

Every one of the 696 parcels is a **closed** mesh — zero open boundary edges, so
each encloses a real volume (median 0.80 mL, max 3.75 mL). Overlap is a true
volume intersection in mL. `atlas.py` voxelises the parcels once onto the MNI
grid and caches the result; intersection is then voxel counting.

**The parcels cover about half the brain.** All 696 total ~656 mL against a
1200–1500 mL brain; deep white matter is not parcellated. Only 27–32% of these
tumours lands in parcelled tissue at all — correct behaviour, not a failure.
The bulk of each lesion sits in white matter the atlas cannot name, which is the
strongest argument for obtaining the tract bundle below.

The grid is 1 mm, so a 0.08 mL parcel is ~80 voxels and its percentage is coarse.

## White matter tracts — needs a data bundle

`TractNames.txt` lists 92 tracts and the `Tract` API includes
`get_start_and_end_parcels()`, which answers "where does this pathway lead". The
geometry is not bundled with the pip package:

```
FileNotFoundError: Tract Path Does Not Exist, please initialize tract with root directory
```

It expects `WhiteMatterTracts/<name>/polydata.vtp` under a separately
distributed data root. `YaleBrainAtlas` is Ryan Bose-Roy's library
(github.com/RyanSBRoy/YaleBrainAtlas), so ask him for that directory.

## Data provenance

Inputs come from a public GitHub-release mirror of TCIA (`data-nih/tcia`, tag
`upenn-gbm`); the official NBIA/Aspera route does not script cleanly. Subject
mapping verified against `participants.tsv`: `sub-NNN` ↔ `UPENN-GBM-00NNN_11`,
`sub-NNN_2` ↔ `_21`.

It is not an official TCIA endpoint, and the volumes are 8-bit. Re-pull from
TCIA before publishing anything that rests on them.

Cite: Bakas et al., *The University of Pennsylvania glioblastoma (UPenn-GBM)
cohort*, Scientific Data 9, 453 (2022).
