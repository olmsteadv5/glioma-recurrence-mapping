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
(SRI24 atlas space). They are **not** skull-stripped — the mirror ships the
`images_structural_unstripped` variant, at ~2650-2930 mL of foreground where a
stripped brain is 1200-1600 mL. Run `scripts/skullstrip.py` before anything
else; it matters a lot (see the results section).

Three caveats:

1. **This is not an official TCIA endpoint.** It is an unaffiliated account
   with no institutional backing. Fine for a pilot; re-pull from TCIA before
   anything leaves the lab.
2. **They arrive unstripped.** `scripts/skullstrip.py` (FreeSurfer SynthStrip)
   handles it, computing one mask on the T1 and applying it to all four
   modalities so they cannot disagree about which voxels exist.
3. **The mirrored volumes are 8-bit (uint8, 0–255).** Whether that matches the
   TCIA originals or is a quantisation introduced by the mirror has not been
   confirmed. BraTS models z-score normalise per modality, so this is unlikely
   to change results much, but it should be checked against a TCIA original
   before any published number rests on it.

Subject id mapping, verified against `participants.tsv`:
`sub-NNN` ↔ `UPENN-GBM-00NNN_11` (baseline), `sub-NNN_2` ↔ `..._21` (later scan).

Cite: Bakas et al., *The University of Pennsylvania glioblastoma (UPenn-GBM)
cohort*, Scientific Data 9, 453 (2022).

## Results

`BraTS25_2` on CPU, `--jobs 1`, skull-stripped inputs. Four subjects, ~3 min
each.

| Subject | WT mL | TC mL | ET mL | Dice WT | Dice TC | Dice ET |
|---|---|---|---|---|---|---|
| UPENN-GBM-00002_11 | 191.1 | 44.5 | 29.6 | 0.919 | 0.958 | 0.900 |
| UPENN-GBM-00036_11 | 166.6 | 40.9 | 20.8 | — | — | — |
| UPENN-GBM-00042_11 | 103.1 | 72.7 | 43.8 | — | — | — |
| UPENN-GBM-00051_11 |  35.5 | 14.7 |  7.8 | — | — | — |

Only subject 2 has published reference labels; the other three are the subjects
with existing tractography (`~/penn_gbm_tractography/fib_files/`).

### Skull-stripping is not optional

The first run of this pipeline used the mirror's images as downloaded, which are
unstripped. Same algorithm, same subjects, only the stripping differs:

| Dice | unstripped | skull-stripped |
|---|---|---|
| WT | 0.757 | **0.919** |
| TC | 0.882 | **0.958** |
| ET | 0.812 | **0.900** |

Predicted whole-tumour volume went 308.8 → 191.1 mL against a 211.8 mL
reference. With skull attached the model labelled non-brain tissue as oedema;
per-label volume ratios are now 0.97 NETC / 0.87 SNFH / 1.06 ET.

The lesson worth keeping: BraTS algorithms assume skull-stripped input and fail
*quietly* when they do not get it — plausible-looking output, quietly wrong.
`verify_inputs.py` now checks absolute brain volume against a 2000 mL ceiling,
which is the test that actually catches this. A fraction-of-grid test does not:
an unstripped head is ~33% of the 240×240×155 grid, inside any plausible band
for a stripped brain.

### Visual QC

All four overlays (`outputs/BraTS25_2/qc/`) are anatomically sound: enhancing
label on the enhancing rim, necrosis filling the core, oedema on FLAIR-bright
tissue. For subject 2 the prediction and the expert reference are now near
indistinguishable at the tumour-bearing slice.


## Mapping the tumour onto the Yale Brain Atlas

Puts the segmentation into the YBA's coordinate frame, then asks which parcels
and white matter tracts it involves.

```bash
# 1. SRI24 -> MNI152 (the YBA reference space), ~1 min/subject
.venv/bin/python scripts/register_to_mni.py

# 2 & 3 need the YaleBrainAtlas package, which lives in the other venv
YBA=/home/vilem/Projects/YBAParcelVisualizer/.venv/bin/python

# 2. which parcels / tracts does the tumour hit
$YBA scripts/tumor_vs_yba.py

# 3. interactive 3D: solid tumour, involved parcels, faint atlas shell
$YBA scripts/visualize_yba.py --shell-opacity 0.2
```

### Why MNI

The YBA is defined in MNI152: its surface spans x ±72.6, y −107.6..73.7,
z −52.4..82.5 with centroid (0, −17, 16), matching the MNI152 1mm template's
(−1, −17, 19). BraTS output is on the SRI24 grid, so it has to be moved.

### Registration with cost-function masking

A glioma displaces healthy tissue. A deformable registration driven by image
similarity will squash or stretch the lesion to make the patient resemble the
template, corrupting the warp exactly where it matters.

`register_to_mni.py` excludes the lesion from the similarity metric. greedy
applies masks to the *fixed* image (`-gm`; its `-mm` moving-image mask errors
out in the ITK-SNAP build here), and the fixed image is the template — so the
mask has to be in MNI space, which we do not have until the tumour is already
there. Hence two passes:

1. register unmasked, push the tumour into MNI to find roughly where it lands
2. build an MNI-space mask excluding that region, re-register with `-gm`

Masking the reference-space lesion is the same convention FSL uses. It is a
mitigation, not a cure — tissue displaced by a large lesion is still displaced,
so treat parcel assignments near the tumour rim as approximate.

**Measure volumes in native space, not MNI.** Warping into a healthy template
squeezes out the mass effect: subject 36's tumour is 234.5 mL natively and
176.9 mL in MNI. Use MNI for *where*, native for *how much*.

### What the parcel overlap actually measures

The YBA is a **surface** atlas — ~63.8k labelled vertices, no volumetric label
map. So overlap is computed by sampling the tumour at each atlas vertex and
aggregating per parcel, which gives the share of each parcel's *surface* inside
the tumour. For cortical parcels, which are sheets, that is close to volume
share. Deep structures are represented by their boundary surface, so a tumour
sitting inside one without reaching its edge is under-counted. Raw vertex counts
are reported alongside the fraction so this stays visible.

### White matter tracts — needs a data bundle

`TractNames.txt` lists 92 tracts and the `Tract` API includes
`get_start_and_end_parcels()`, which answers "where does this pathway lead".
But the geometry is not bundled with the pip package:

```
<Tract 'ArcuateFasciculusL' | Fibers: Not Loaded>
FileNotFoundError: Tract Path Does Not Exist, please initialize tract with root directory
```

The package expects `WhiteMatterTracts/<name>/polydata.vtp` under a separately
distributed data root. `YaleBrainAtlas` is Ryan Bose-Roy's library
(github.com/RyanSBRoy/YaleBrainAtlas), so **ask Ryan for that directory**. The
tract code here is written against the documented API and stays dormant until
the bundle is present.
