# Cohort and recurrence overlap

The baseline work: which patients are in the study, how we know the list is
complete, and how much of each patient's later tumor overlaps the first.

- `measure.py` computes the overlap. `overlap_csv.py` writes
  `outputs/results/<space>/tables/overlap_review.csv` from it, one row per patient.
- `evidence/` holds what the cohort was checked against: the mirror's file
  list and sizes, the release's participant table, the TCIA study-date query,
  and `audit.py`, which re-checks every file of every patient.
- `report/` is the summary page sent with the CSV
  (https://claude.ai/artifact/99SBHUXymP8NQLc27URqQd) and the numbers behind it.

Copied from the planning notes when the target planning was archived on
2026-09-28; the originals stay in `archive/target-planning/DECISIONS.md`.

## The cohort, and how we know it is complete

**41 patient pairs.** Three independent sources agree on exactly the same 41:

- the release's own participants table (`inputs/participants.tsv`): 671
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

**One segmentation model for every scan.** All 82 scans are segmented by the
BraTS 2025 adult glioma **winning** entry (BraTS25_1),
`brainles/brats25_gli_aia-bots`, image digest
`sha256:3b531f596b25aec240c2b635caab796263c4af37bc5e3045261c87446c3b33ac`, run
2026-10-02 on Yale Bouchet GPUs (Slurm jobs 28160680-28160683) since it needs
CUDA. Its PyTorch has no kernels for Blackwell cards, so it has to be kept to
A40, A5000, L40S or RTX 5000 Ada nodes. Each scan was named
`BraTS-GLI-<number><timepoint>-100`, as the brats package names everything, so
the model applies its post-treatment cleanup thresholds to both timepoints.

Before that, every scan was segmented on CPU by the **second-place** entry,
`brainles/brats25_gli_ecnu`, digest
`sha256:c2a27297deab0638296e8b90569d19f7dfda1519ed116093cb2602fa8f1f1596`. Those
segmentations and every result made from them are kept in `outputs/archive/brats25_2/`. The
two agree closely: median Dice per scan 0.99 for whole tumor and 0.98 for
enhancing tumor at both timepoints, lowest 0.93 and 0.78. They differ mainly in
the cavity, which the winner labels on 16 follow-ups instead of 11, taking some
voxels the second-place model called non-enhancing core.

**Download integrity.** Every input file's size matches the mirror's recorded
asset size to the byte.

**Registration between timepoints**, all pairs: brain Dice 0.978 to 0.990,
median 0.988.

**Found in passing, not yet used:** the mirror carries reconstructed diffusion
imaging (DSI Studio GQI and QSDR) for 493 baselines and 38 follow-ups, and the
participants table scores each follow-up on a 1 to 6 `PsP_TP_score`, the
pathology of the second resection (see below).

## Which compartment counts as the recurrence

**Open, and for the supervisor to settle before a model is fitted.** The
compartments of the later tumor behave very differently (all 41 pairs, patient
space, from `overlap_review.csv` and `capture_curves.csv`):

| later compartment | median share on the first tumor | median share inside the NRG 46 Gy volume | margin the median patient needs for 95% | pattern against the 60 Gy volume |
|---|---|---|---|---|
| whole (WT) | 0.46 | 0.97 | 18 mm | 6 central, 10 in-field, 25 marginal |
| core (TC) | 0.35 | 1.00 | 4 mm | 26 central, 9 in-field, 4 marginal, 1 distant |
| enhancing (ET) | 0.20 | 1.00 | 5 mm | 24 central, 10 in-field, 5 marginal, 1 distant |

One patient has no enhancement or core at follow-up. Clinical recurrence is
conventionally read from new **enhancement**, and on that definition a 2 cm
margin already contains essentially all of it in most patients. If the endpoint
is ET, coverage has almost no room to improve and the claim becomes volume
spared at equal coverage. If it is WT, the method is being scored on where FLAIR
signal spreads, which is partly edema and partly infiltration, and the result has
to be described that way. So far every tract effect is in WT and none in ET.

**Caveats on the table itself.** `_11` is pre-operative and `_21` is simply the
next study, so a later scan is not always a recurrence: 00055 at 31 days is
untreated growth. Treatment status is not recorded and can only be inferred
from `kept`. The 60 Gy proxy is 2 cm from the core with no
PTV margin, so it is smaller than a real 95% isodose and the pattern counts lean
toward marginal.

## Registration between timepoints is not a problem

Checked, because a target and the recurrence it is scored against reach MNI
through two separate warps. Brain dice 0.987 to 0.989 across five subjects, mean
residual deformation 0.92 to 1.38 mm, 95th percentile 3.2 to 4.1 mm. The mean is
well inside a voxel. The tail is comparable to the effects being measured, so it
is worth stating as a limitation, but it is not corrupting anything.

**In patient space** each pre-op scan is fitted rigidly onto its own follow-up
T1 (`scan/align.py`, greedy, NCC, both tumors masked out of the fit), so every
pair is measured on the scan the tumor came back on, as Dr. Munbodh asked. Brain
Dice across the 41 goes from a median 0.978 as shipped to 0.981. Tumor volumes
survive the resampling to within 1%. The lowest after fitting, 00301 at 0.930
and 00130 at 0.957, were looked at: midline and ventricles line up, and what is
left is real change between the scans (pre-op swelling, a cyst that is gone), which
a rigid fit should not undo. NMI was tried first and swung 00130 off by 21
degrees, so the stage now refuses any fit that ends clearly worse than the start.
The shipped grid is itself rigid: the release put every scan on SRI24 with
"rigidly (6 degrees of freedom)" registration, so native volumes are the
patient's own.

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

*This section was written with the second-place (25_2) segmentations. The winner
labels a cavity on 16 of the 41 follow-ups, 9 of them over 3 mL, and catches
00314's; the numbers below have not been redone with it.*

**The model labels a cavity on only 11 of the 41 follow-ups** (none on the
pre-op scans), though every patient was resected. All 41 were looked at, one
axial slice through the original tumor site on T1-Gd, FLAIR and T2
(`outputs/results/qc/cavity_review/`, made by `evidence/cavity_sheets.py`):

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


## Walk-down target

`walkdown.py` builds a target that follows white matter instead of growing
evenly, for both NRG phases: from the whole tumor (the 46 Gy volume) and from the
core (the 60 Gy boost, kept inside the first at the same distance). Every atlas
streamline with a point in the tumor is followed d mm along its own path from
each place it leaves; one convex hull is wrapped around the tumor's edge and all
those end points. Walks stop at d, at the end of the streamline, at the brain
edge, back inside the tumor, or where a plan may not go. Nothing goes outside the
brain. The tumor's own hemisphere is open; across the midline a voxel may join a
target grown d mm only if it is within d of the part of that phase's tumor
already across (the whole tumor for 46 Gy, the core for 60 Gy), so nothing grows
over the falx from the home side. Less than 0.1 mL across counts as registration
slop, not crossing; 11 of the 41 cross by that measure. The plain margin and the same-size ball obey
the same rule, so all three are compared on the same terms. There is no minimum
margin.

The streamlines are the HCP-1065 tract atlas [yeh22]: 77 bundles once the
cranial nerves are dropped, 479,804 streamlines, of which a fixed 1 in 10 of
every bundle is used (47,980), resampled at 1 mm and registered from the
atlas's ICBM152 template onto our MNI brain (`hcp/streamlines.py`).

It is built at every whole mm from 0 to 40, so nothing between two samples goes
unseen; 20 mm stays the reference, and is the target the viewer draws. Each
voxel stores the shortest walk at which it joins the target, and the target at d
is every voxel reached by d or less, so it only grows with d. Even at d = 0 the
hull is bigger than the tumor, since it fills the tumor's dents, which is why the
ball of the same size, not the margin of the same distance, is the fair control.

Built in MNI, where the tracts are, and carried into each follow-up scan along
the pre-op scan's own road (its inverse warp, then the rigid fit), which puts an
MNI copy of the pre-op tumor back on top of the patient-space one at Dice 0.988.
Checked on a made-up 15 mm sphere with straight tracts in 2000 directions: the
hull came out 176.9 mL against the 179.6 mL of a perfect 35 mm sphere. A tract
bent into a 10 mm circle ends 16.8 mm away after 20 mm of path, as it should.

Patient space, at 20 mm:

| | 46 Gy walk | NRG 2 cm | 60 Gy walk | NRG 2 cm |
|---|---|---|---|---|
| brain beyond the tumor, median | 167 mL | 289 mL | 76 mL | 180 mL |
| enhancing recurrence caught, median | 99.0% | 100.0% | 88.5% | 99.8% |
| whole recurrence caught, median | 83.3% | 96.9% | 48.7% | 74.1% |

Against a ball of exactly its size, the walk catches the same share of the
enhancing recurrence (at 20 mm, 46 Gy: better in 11, worse in 17, p = 0.72) and
more of the whole recurrence: +1.5 points at 10 mm and at 20 mm for 46 Gy
(better in 27, worse in 12, p < 0.001), and +3.0 at 20 mm for 60 Gy (37 against
4). The median patient's 46 Gy walk-down first catches 95% of the enhancing
recurrence at 6 mm, against 5 mm for the plain margin and 4 mm for the ball; the
60 Gy walk never gets there. MNI agrees (46 Gy, whole recurrence: 27 better, 13
worse; 60 Gy: 37 against 3).

The 3 mm skim rule (a tract leaving and re-entering within 3 mm is not an exit)
was checked with the earlier tract set: run with it off, no patient's target
moved by even 0.1 mL at any distance, because a skimming tract's end points
already sit inside the hull around the tumor's edge. Not yet rechecked with
the HCP-1065 streamlines.

Past 20 mm the walk-down keeps growing slowly, since streamlines end and the
midline and brain edge stop walks: at 40 mm it still spends less brain than the
2 cm volume (46 Gy: median 286 against 289 mL, by when it catches 97.3% of the
whole recurrence against the 2 cm volume's 96.9%). Its edge over the same-size
ball on the whole recurrence peaks near 14 mm (+1.8 points) and is gone by 35 mm
for 46 Gy, while for 60 Gy it holds at about +3 points out to 40 mm. At equal
coverage (the shortest walk catching at least as much as the 2 cm volume, in
`walkdown_matched.csv`), only some patients get there within 40 mm: for 46 Gy,
20 of 40 for the enhancing recurrence, saving a median 191 mL each, and 20 of 41
for the whole recurrence, saving 57 mL; for 60 Gy, 13 of 40 (95 mL) and 25 of 41
(25 mL). The others never catch as much, so those medians describe the patients
whose tumors suit the walk-down, not everyone.

**How many streamlines.** A hull is set by its outermost points, so keeping more
streamlines makes it bigger. `streamline_sampling.py` rebuilds every patient at
1/20, 1/10, 1/5, 1/2 and all of the atlas, each share inside the next, plus two
more random draws at 1/20 and 1/10 (MNI, 46 Gy, 20 mm):

| share kept | 1/20 | 1/10 (used) | 1/5 | 1/2 | all |
|---|---|---|---|---|---|
| brain beyond the tumor, median | 179 mL | 184 mL | 188 mL | 192 mL | 194 mL |
| against all, median patient | -9% | -6% | -4% | -2% | 0 |
| against all, worst patient | -21% | -17% | -12% | -8% | 0 |
| whole recurrence, walk minus same-size ball | +3.2 | +3.0 | +3.0 | +3.2 | +3.1 |

Absolute volumes therefore run about 6% below the full atlas, while the
comparison with the same-size ball, which grows with the walk-down, does not
move; repeat draws at the same share land within about 2 mL. The 60 Gy target
behaves the same way (-9% at 1/10).

Open: the midline rule and its 0.1 mL floor are ours, built on the NRG falx rule
rather than taken from a paper. 00354's pre-op scan
has no core at all, so its 60 Gy walk falls back to the whole tumor; under NRG it
would have no boost volume.

## Recurrence on and off the tracts

`tract_shell.py` asks the tract question without building any target. In 1 mm
shells around the pre-op whole tumor, white matter a streamline through the
tumor passes is compared with white matter at the same distance that no such
streamline passes (SynthSeg white matter only, since tracts only run there and
so does swelling; MNI, where the atlas is). Streamlines through the tumor touch
much of the nearby white matter (median 75% of the first shell, 36% at 20 mm),
but the smallest off-tract shell in any patient still holds 338 voxels.

The whole recurrence is more likely on a tract at every distance out to 20 mm:
about 8 percentage points in the median patient, and higher on the tract in 32
of 41 patients at 1 mm, 37 at 5 mm and 30 at 15 mm. Part of this may be
swelling following the tracts rather than tumor.

The enhancing recurrence shows no difference: higher on the tract in 23 of 40
patients at 1 mm, 19 at 5 mm and 10 at 15 mm, median difference about 0. (An
earlier, sparser tract set showed a small on-tract excess in the first 5 mm; it
did not survive the change to the HCP-1065 streamlines.)
