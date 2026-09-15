# Brain tumour segmentation on UPenn-GBM

Runs the BraTS competition algorithms (https://github.com/BrainLesion/BraTS) on
glioma MRI, then works out which Yale Brain Atlas parcels each tumour sits in
and which white matter tracts run through it.

```bash
./main.py 36 42 51
```

That downloads the scans, skull-strips them, checks them, segments, scores
against reference labels if the subject has any, registers into MNI, intersects
with the atlas and opens a 3D view. Everything reruns each time except the
downloads and the cached atlas files.

To reopen a view without redoing the work:

```python
import json, view
results = json.load(open("outputs/results.json"))
view.run("UPENN-GBM-00036_11", results["UPENN-GBM-00036_11"]["atlas"])
```

## Setup

Everything runs on one interpreter:

```
/home/vilem/Projects/YBAParcelVisualizer/.venv/bin/python
```

It has brats, nibabel, scipy, pyvista, trimesh and plotly. Docker needs to be
running and you need to be in the docker group. The algorithm image is about
30 GB and pulls itself the first time you run.

## Files

```
main.py        runs everything
fetch.py       download scans
skullstrip.py  SynthStrip
verify.py      input checks
segment.py     BraTS in Docker
evaluate.py    volumes and Dice
register.py    SRI24 to MNI152
atlas.py       atlas volume, meshes, parcel overlap
tracts.py      tract centrelines
view.py        3D view
paths.py       shared paths

inputs/<SUBJECT>/         raw scans, never modified
inputs/reference_seg/     expert labels, 147 of 630 subjects have them
inputs/atlas/             official Yale Brain Atlas data

outputs/<SUBJECT>/stripped/           skull-stripped scans
outputs/<SUBJECT>/seg.nii.gz          segmentation, SRI24 space
outputs/<SUBJECT>/seg_in_mni.nii.gz   same thing in atlas space
outputs/atlas/                        cached atlas derivatives
outputs/results.json                  volumes, Dice, parcel overlap
```

## Results so far

BraTS25_2 on CPU, roughly 3 minutes a subject.

```
subject                 WT     TC     ET    Dice WT/TC/ET
UPENN-GBM-00002_11   191.1   44.5   29.6    0.919 / 0.958 / 0.900
UPENN-GBM-00036_11   166.6   40.9   20.8    no reference labels
UPENN-GBM-00042_11   103.1   72.7   43.8    no reference labels
UPENN-GBM-00051_11    35.5   14.7    7.8    no reference labels
```

## Skull stripping is not optional

The mirror ships the unstripped version of UPenn-GBM. BraTS assumes stripped
input and quietly does a worse job without it. Same algorithm, same subjects,
stripping the only difference:

```
        unstripped   stripped
WT         0.757      0.919
TC         0.882      0.958
ET         0.812      0.900
```

Predicted whole tumour volume went from 308.8 mL to 191.1 mL against a 211.8 mL
reference, and the spurious resection cavity labels on pre-op scans went away.

verify.py checks brain volume in mL against a 2000 mL ceiling. A percentage of
the grid does not catch this, since an unstripped head is about 33% of a
240x240x155 grid, which is inside any sensible range for a stripped brain. That
mistake cost me a full run.

## Modality naming

BraTS names things differently to UPenn-GBM:

```
t1c   post-contrast T1   ->  _T1GD
t1n   native T1          ->  _T1
t2f   T2-FLAIR           ->  _FLAIR
t2w   T2                 ->  _T2
```

Swapping t1c and t1n does not throw an error, it just ruins the enhancing
tumour label. ET Dice is the thing that tells you.

## Algorithm

BraTS25_1 won the 2025 challenge but only runs on CUDA. segment.py uses it when
there is a GPU and falls back to BraTS25_2, the best CPU one, otherwise. On a
GPU machine you get the winner without changing anything.

## Registration

The atlas lives in MNI152 and BraTS output is on the SRI24 grid, so the tumour
has to be moved. The problem is that a glioma pushes tissue around, and a
registration driven by image similarity will happily squash the lesion to make
the patient look more like the template, which wrecks the warp exactly where you
care about it. register.py excludes the lesion from the deformable metric with
greedy's -mm flag, which is standard practice going back to Brett et al. 2001.

Two things about greedy that are not documented anywhere:

- -mm crashes in affine mode. The affine stage runs unmasked instead, which
  costs almost nothing since a 12 parameter transform cannot warp locally.
- -mm only works with -m NCC. The help text mentions a WNCC metric that -m does
  not accept, so it must switch to it internally when you pass a mask. Every
  attempt with NMI failed.

An earlier version worked around the crash with two passes and a mask in
template space. The single pass version squashes the lesion noticeably less:
Jacobian ratio of lesion to healthy tissue 1.365 against 1.807, and tumour
volume 151.8 mL against 119.4 mL where native is 166.6 mL.

Masking stops the lesion corrupting the warp but does not undo mass effect, so
parcel assignments near the tumour edge are approximate. Measure volumes in
native space, not MNI.

## Parcel overlap

The official 696 parcel volume gets resampled onto the MNI grid, then overlap is
voxel counting.

The parcels only cover grey matter, about 731 mL against a 1200 to 1500 mL
brain, so only around 30% of each tumour lands in a parcel at all. That is
expected, not a bug. The rest is in white matter.

## Tracts

tracts.py uses the official top 5000 tract mesh and draws the tracts that pass
through the tumour or one of the involved parcels.

The file is a Blender tube mesh rather than line geometry, so it gets reduced to
centrelines once and cached. Vertices are Y-up and map to atlas space as
(x, -z, y). Connected components give 3001 bundles, each averaged down to a line
in 4 mm bins along its long axis.

```
                      through tumour   plus parcels   of 3001
UPENN-GBM-00002_11               915           1053
UPENN-GBM-00036_11               716            831
UPENN-GBM-00051_11               356            369
```

The mesh is one unnamed object, so these are tract paths, not named bundles like
the arcuate fasciculus. Named bundles are not in the Yale data at all; the names
in Ryan's wrapper look like DSI Studio's tractography atlas. Pass
show_tracts=False to view.run to hide the lines.

## Atlas data

All from github.com/YaleBrainAtlas/YaleBrainAtlas, downloaded into inputs/atlas
on first run.

```
YBA_696.nii                   parcel volume
YBA_696_LUT.txt               label index to name
YBA_696_parcel_dict.csv       long names, lobe, gyrus, colours
YBA_696.glb                   per parcel surfaces
top5k_tracts_decimated.obj    tract mesh
```

The GLB has x and z negated relative to the parcel dictionary. atlas.GLB_FLIP
fixes it, checked against the dictionary centroids to 0.00 mm.

## Where the scans come from

A public GitHub mirror of TCIA (data-nih/tcia, tag upenn-gbm), because the
official NBIA and Aspera route does not script cleanly. The subject mapping is
verified against participants.tsv: sub-NNN is UPENN-GBM-00NNN_11 and sub-NNN_2
is the _21 timepoint.

It is not an official TCIA endpoint and the volumes are 8 bit, so re-pull from
TCIA before publishing anything that depends on them.

Cite Bakas et al., The University of Pennsylvania glioblastoma (UPenn-GBM)
cohort, Scientific Data 9, 453 (2022).
