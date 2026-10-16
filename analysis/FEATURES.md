# Per-voxel dataset

The plan for one dataset covering the 41 pairs: every brain voxel, every
variable that can be measured from the pre-op scan, and the recurrence labels
from the follow-up. Built once, checked, documented, and read by any later
analysis. Nothing here is built yet.

## Raw only, rules applied later

The dataset stores only measurements that are true of the patient and the
atlas. No rule, threshold or target from earlier tests goes into it: no midline
rule or 0.1 mL floor, no 20 mm cutoff, convex hull or skim rule, no 1 in 10
streamline sampling, no intensity normalization, no NRG or walk-down volumes.
An analysis applies any of those when it reads the data, so they can change
without rebuilding anything.

The only processed inputs, each recorded in `spec.json`: the tumor segmentation
(BraTS 2025 winner), the registrations (each scan onto MNI, the HCP-1065
template onto MNI), SynthSeg's tissue labels and the lobe map.

## Steps

1. **Diffusion and perfusion.** The UPenn-GBM release on TCIA includes
   diffusion (FA, MD, AD, RD) and perfusion (rCBV, peak height, signal recovery)
   maps already on each patient's grid; the GitHub mirror does not. Download
   them, record per patient what exists in `coverage.csv` (the release paper
   puts diffusion at about 88% of scans and perfusion at about 80%), and check
   two or three against their FLAIR.
2. **Space rules.**
   - One grid: MNI, 1 mm, the whole brain. No distance cutoff: 40 mm would miss
     recurrence in 13 of the 41 patients, and the farthest reaches 86 mm.
   - Each image moves into MNI once; labels with label interpolation, values
     with linear.
   - Distances, gradients and textures are computed after the move, on the MNI
     grid.
   - Inputs come from the pre-op scan only; the follow-up gives the labels and
     nothing else.
   - 32-bit floats for measurements, small integers for categories.
3. **Variables.** About 67 per voxel; ×2 means computed from both the whole
   tumor and the core, the two NRG phases' starting points.

| group | variables | source |
|---|---|---|
| tumor geometry | straight-line distance to the tumor edge (×2) · inside the tumor's own convex hull (×2) · along the brain's depth layers or across them, from the tumor (×2) · curvature of the nearest tumor edge (×2) · how far the FLAIR halo reaches from the core in this voxel's direction · pre-op compartment · distance to the enhancing edge · distance to the necrosis edge | segmentation |
| midline | signed distance to the midline plane · on the tumor's main side or not · distance to any tumor already across (×2) | segmentation, template |
| anatomy | depth below the brain surface · distance to the ventricles · distance to the lateral ventricle wall · lobe | template, SynthSeg, lobe map |
| tissue | SynthSeg class · distance to the gray/white border · distance to sulcal CSF · white matter share within 5 mm | SynthSeg |
| tracts, all 479,804 streamlines | shortest distance along any streamline through the tumor, no hull, stops or limit (×2) · streamlines through the tumor passing the voxel (×2) · their split into association, commissural and projection (×2) · alignment of the local fibers with the direction away from the tumor (×2) · total streamline density · number of bundles crossing · distance to the nearest streamline end · highest bundle probability | HCP-1065 |
| connectivity | share of the parcel's connectivity running into the tumor's parcels · share of the parcel the tumor holds | Yale atlas |
| structural intensity | T1, T1-Gd, T2, FLAIR · FLAIR gradient · local variability of FLAIR and T1-Gd | pre-op MRI |
| diffusion | FA · MD · AD · RD | TCIA |
| perfusion | rCBV · peak height · signal recovery | TCIA |

   Per patient, in `patient.json`: compartment volumes, volume across the
   midline, tumor depth, lobe, ventricle contact; each sequence's mean and
   spread in normal-appearing white matter, so any normalization can be applied
   later; which maps exist; and, marked optional, age, sex, IDH, MGMT and extent
   of resection.

   Labels, from the follow-up only: enhancing, core and whole recurrence, a flag
   for recurrence outside the pre-op tumor, and the pathology score.

   Left out: NRG, walk-down and ball targets (analyses, built later from the raw
   columns); raw coordinates, parcel IDs and bundle IDs (too many categories for
   41 patients); parcel function terms (not about spread); distance to the skull
   (repeats depth); days between scans and survival (not known when planning).
4. **Files.**

```
outputs/features/
  spec.json              every variable: name, units, method, reference, source, code version
  coverage.csv           per patient: which maps exist, anything wrong
  UPENN-GBM-<n>/
    voxels.npz           one row per brain voxel (~2.1 M): its place on the MNI grid,
                         every variable, the labels
    patient.json         the patient-level values
```

   About 0.55 GB a patient, 23 GB for the 41.
5. **Build.** `analysis/features.py`, written against `spec.json`, reusing the
   distance maps, tract lookups and registrations already here but none of the
   walk-down's rules. One patient at a time, each saved as it finishes so a
   rerun skips it.
6. **Check before use.**
   - Rebuild known results from the raw columns: the plain margin from the
     distance column, the on and off tract result from the streamline count,
     and the NRG volumes by applying the midline rule to the raw midline
     columns. They must match the existing numbers.
   - Look at the QC images for at least five patients, 00160 (crosses the
     midline) and 00354 (no core) among them.
   - Ranges: FA within 0 to 1, distances not negative outside the tumor, a
     missing map marked missing rather than filled with zeros.
   - No input column read from a follow-up scan.

The same pipeline can later run on the other ~590 UPenn pre-op patients, for
variables without labels, once they are segmented on Bouchet.

## Running it on this laptop

12 cores (24 threads), 22 GB of memory. The run is fenced so it can slow down
but never take the desktop with it:

- the whole job sits in its own cgroup (`systemd-run --user`), throttled above
  11 GB and stopped at 13 GB, so a memory overrun ends the job and nothing else;
  each worker also keeps its own cap and the free memory check before it starts
- lowest CPU priority (`nice 19`, low CPU weight) and idle disk priority, with
  at most three workers of about four threads each
- sleep and lid suspend blocked while it runs; before each patient it waits
  unless the laptop is on mains power
- one pilot patient first, to measure memory and time and set the number of
  workers from that
- pause, resume and stop from one command each; finished patients stay saved

Estimated: the download 30 to 90 minutes, the pilot about 10, the other 40
patients 1.5 to 2.5 hours three at a time.
