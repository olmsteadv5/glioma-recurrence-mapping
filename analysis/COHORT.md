# Cohort and recurrence overlap

The baseline work: which patients are in the study, how we know the list is
complete, and how much of each patient's later tumor overlaps the first.

- `measure.py` computes the overlap. `overlap_csv.py` writes
  `outputs/overlap_review.csv` from it, one row per patient.
- `evidence/` holds what the cohort was checked against: the mirror's file
  list and sizes, the release's participant table, the TCIA study-date query,
  and `audit.py`, which re-checks every file of every patient.
- `report/` is the summary page sent with the CSV
  (https://claude.ai/artifact/99SBHUXymP8NQLc27URqQd) and the numbers behind it.

Copied from the planning notes when the target planning was archived on
2026-09-28; the originals stay in `archive/target-planning/DECISIONS.md`.

## The cohort, and how we know it is complete

**41 patient pairs.** Three independent sources agree on exactly the same 41:

- the release's own participants table (`outputs/participants.tsv`): 671
  scans, 611 baselines (`_11`), 60 follow-ups (`_21`), 41 patients with both
- the public mirror's file list: 41 patients with a `sub-NNN_2` scan, all four
  sequences present for both timepoints, no third timepoints
- the TCIA API: 43 patients with two study dates, the 41 plus two below

Excluded, with reasons:

- **00156 and 00542.** Two TCIA study dates, 1 and 2 days apart: the same tumor
  scanned twice, not a follow-up. The curated release does not include a
  second timepoint for either.
- **00612 to 00630, 19 patients.** Only a follow-up scan exists in the
  collection; there is no baseline to compare against. The mirror names that
  single scan `sub-NNN` with no `_2`, so the pipeline would label it `_11`;
  none of them are ever fetched as a pair.

**00042 and 00051 were missed at first and are included.** Both were processed
early, first scan only, and the to-do list treated a patient with an output
folder as done. Found by checking against the mirror rather than our own lists.

**The interval comes from the release, not from TCIA dates.** 00130 has an
extra single-series study 11 days before its curated follow-up, so taking the
second TCIA date gave 433 days where the release says 444. `measure.interval`
now reads `Time_since_baseline_preop` from the participants table.

**One segmentation model for every scan.** All scans were segmented on CPU by
the BraTS 2025 adult glioma **second-place** entry, `brainles/brats25_gli_ecnu`,
image digest `sha256:c2a27297deab0638296e8b90569d19f7dfda1519ed116093cb2602fa8f1f1596`,
built 2025-08-07 and never re-pulled. The winning entry needs CUDA. Checked
against every scan's own segmentation log, not the summary file, since four
early scans predate the pipeline recording which algorithm it used.

**Download integrity.** Every input file's size matches the mirror's recorded
asset size to the byte.

**Registration between timepoints**, all pairs: brain Dice 0.978 to 0.990,
median 0.988.

**Found in passing, not yet used:** the mirror carries reconstructed diffusion
imaging (DSI Studio GQI and QSDR) for 493 baselines and 38 follow-ups, and the
participants table scores each follow-up on a 1 to 6 `PsP_TP_score`, the
pathology of the second resection (see below).

## Which compartment counts as the recurrence

*The table in this section is from the first 21 pairs. The full 41 are in
`outputs/overlap_review.csv` and on the summary page.*

**Open, and for the supervisor to settle before the model goes further.**

`planeval.coverage` scores a target against the **whole** later segmentation,
`data > 0`: enhancing tumor, necrosis, and FLAIR hyperintensity together. Every
coverage number so far, the corridor null included, is therefore mostly a
measure of catching edema.

`measure.overlap` on the first 21 pairs shows the compartments behave very
differently:

| later compartment | median share on the first tumor | median share within 2 cm (60 Gy proxy) | margin covering 95% of it | pattern |
|---|---|---|---|---|
| whole (WT) | 0.40 | 0.72 | 20.3 mm | 3 central, 5 in-field, 13 marginal |
| core (TC) | 0.36 | 0.99 | 4.9 mm | 14 central, 3 in-field, 3 marginal |
| enhancing (ET) | 0.19 | 0.99 | 5.7 mm | 14 central, 2 in-field, 4 marginal |

Clinical recurrence is conventionally read from new **enhancement**, and on that
definition a 2 cm margin already contains essentially all of it in most
patients. If the endpoint moves to ET, coverage has almost no room to improve
and the claim becomes volume spared at equal coverage. If it stays on WT, the
method is being scored on where FLAIR signal spreads, which is partly edema and
partly infiltration, and the result has to be described that way.

**Caveats on the table itself.** `_11` is pre-operative and `_21` is simply the
next study, so a later scan is not always a recurrence: 00055 at 31 days is
untreated growth. Treatment status is not recorded and can only be inferred
from `kept`. Volumes are in MNI. The 60 Gy proxy is 2 cm from the core with no
PTV margin, so it is smaller than a real 95% isodose and the pattern counts lean
toward marginal.

## Registration between timepoints is not a problem

Checked, because a target and the recurrence it is scored against reach MNI
through two separate warps. Brain dice 0.987 to 0.989 across five subjects, mean
residual deformation 0.92 to 1.38 mm, 95th percentile 3.2 to 4.1 mm. The mean is
well inside a voxel. The tail is comparable to the effects being measured, so it
is worth stating as a limitation, but it is not corrupting anything.

## What the follow-up scans are, and the resection cavities

From the dataset paper (Bakas et al., Sci Data 2022, PMC9338035), the follow-up
inclusion criteria "comprised 1) initial gross total resection of the tumor core
followed by chemo-radiation, 2) new or increasing enhancement on follow-up MRI,
3) second resection and histopathological tissue evaluation, and 4) acquisition
of all modalities (T1, T1-Gd, T2, T2-FLAIR, DSC, and DTI), within 15 days prior
to the second resection." The
`PsP_TP_score` in `participants.tsv` is the pathology of that second resection:
the share of tissue with malignant features, 1 = <10%, 2 = 10-25%, 3 = 25-50%,
4 = 50-75%, 5 = 75-90%, 6 = >90%. Across our 41: 3, 4, 6, 7, 10 and 11
patients at scores 1 to 6.

Two things in the data do not fit those criteria. Five of the 41 are listed as
partial resections (`GTR_over90percent = N`). And 00055's follow-up is 31 days
after its pre-op scan and looks like an unresected tumor, which leaves no room
for a resection and chemoradiation in between.

**The model labels a cavity on only 11 of the 41 follow-ups** (none on the
pre-op scans), though every patient was resected. All 41 were looked at, one
axial slice through the original tumor site on T1-Gd, FLAIR and T2
(`outputs/figures/cavity_review/`, made by `evidence/cavity_sheets.py`):

- where it labels one, the cavity looks right (00160, 00344, 00130, 00267)
- at least one clear cavity is left unlabeled (00314); unlabeled voxels are in
  none of the three volumes, so that is harmless
- most of the rest show a ring-enhancing lesion at the resection site with a
  dark center, where a cavity with enhancing recurrence and necrotic tumor are
  hard to tell apart on one slice
- 00354's follow-up does not line up with its pre-op scan in their shared
  native frame; the analysis itself runs in MNI, where the pair registers fine

**A missed cavity could only distort the results by being labeled
non-enhancing core.** The enhancing tumor volume never includes that label.
Non-enhancing core is a median 2.9% of the later whole tumor (at most 22.7%)
and 15.5% of the later tumor core. Treating every non-enhancing voxel as cavity
moves the median whole tumor capture at 20 mm from 93.6% to 93.4%, at most 4.0
points in any patient. The core curve could move toward the enhancing curve,
which it already nearly matches.

An intensity test (fluid = ventricle-like on T1, T2 and FLAIR) was tried and
dropped: it called the model's own cavities fluid in only 4 of 11 patients, so
cavity contents here are not ventricle-like enough for it to work.

