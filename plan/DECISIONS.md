# Decisions and open questions

Choices that are not obvious from the code, and questions nobody owns yet. A
decision here was made deliberately and can be revisited; it is not a bug.

Format: what was decided, why, and what would change it.

---

## Anatomy we model rather than observe

### The falx seals the anterior commissure and the fornix

`barriers.falx` is the left/right cerebral interface minus the six corpus
callosum parcels. Every other midline crossing is therefore closed, including
the anterior commissure and the fornix.

Anatomically this is wrong. The real falx has a free inferior edge arching over
the callosum, and nothing below it is dura at all.

**Why we accept it.** The commissure is small, and it only matters if a tumor
comes within `ceiling_mm` of it. Adding a second punch-out without a case that
needs it would be inventing anatomy to fix a problem we have not seen.

**What would change it.** A temporal or basal-frontal tumor where the target
visibly stops at the midline in a place it should not. Look for it before
assuming the falx is right.

### The tentorium seals the incisure

`barriers.tentorium` closes as a ring around the cerebellum and brainstem. The
incisure, where the midbrain passes through, is closed with it.

There is no dura at the incisure. This is a modeling choice, not anatomy, and it
means a corridor cannot descend the corticospinal tract from cerebrum into
brainstem.

**Why we accept it.** The brainstem is a standing avoidance structure at 0 mm
under ESTRO anyway, so a corridor that reached it would be heavily penalized in
any case. The difference between "cannot reach" and "penalized for reaching" is
small here.

**What would change it.** A motor-strip tumor whose recurrence descends into the
brainstem and which we miss. Subject 55 is a motor tumor and is the first place
to look.

---

## Parameters that are not what their citation says

### `wm_gain = 5.0` is a migration rate used as a destination probability

The five-fold figure comes from reaction-diffusion models of glioma growth,
which fit a **diffusion coefficient** — how fast cells *travel* through white
matter versus gray. Our speed field uses it to decide where tumor *ends up*.

Those are different quantities. A cell can migrate quickly along a tract and
settle in the cortex at the end of it. Cells travel on the highway; they do not
live on it.

Evidence that this bites: on subject 00129 the 20 mm neighborhood is 45% cortex,
our target spent 21% of its budget there, and the recurrence we missed was 40%
cortex. The weighting steered the budget away from where the tumor actually went.

**What would change it.** Either a fitted value that comes out near 1, or the
bidding reframe below, which separates the two quantities properly.

### `satellite_ml = 2.0` is a judgement call

A noise floor, not a biological threshold. Registration error and segmentation
speckle produce small spurious fragments; without a floor they would be kept as
satellites. Two millilitres is a sphere about 16 mm across, comfortably larger
than any plausible artifact and above the delivery penumbra.

**What would change it.** Sweeping it. If satellites at 1 mL and 5 mL give the
same coverage, the parameter does not matter and we should say so. Nobody has
run that sweep. A dosimetry figure for the smallest separately plannable volume
would turn it into a constraint rather than a choice.

### `eps_csf = 0.05` is a regularizer, not a model

It should physically be zero — tumor does not invade cerebrospinal fluid. It is
nonzero only because the solver divides by speed and needs the field defined
everywhere. The test is whether the answer moves between 0.01 and 0.1, never
whether 0.05 is "right".

---

## Scope

### The volume budget is a clinical input, not something we optimize

We answer "given N millilitres, what is the best shape". We do not answer "what
is the best N".

**Why.** The 60 Gy prescription is fixed by tumor control probability, and how
much brain may receive it is bounded by normal-tissue constraints on specific
structures — brainstem 54 Gy, chiasm 54 Gy, hippocampus under 9 Gy mean. There
is no published cap on total irradiated volume; a dosimetrist resolves it per
patient.

**What would change it.** The principled version is a trade-off curve: more
volume buys coverage and costs toxicity, and the optimum is where the marginal
exchange turns unfavorable. Our coverage-versus-volume curve is the first half
of that. The second half needs a dose-toxicity model, which needs delivered
dose, which needs a dataset shipping RTDOSE. Burdenko has one.

Until then, report the curve and let the clinician pick the operating point.

---

## An idea worth keeping

### Bidding: separate reachability from desirability

The budget is currently spent greedily by arrival time, nearest first. The speed
field therefore does double duty — it decides both how *reachable* a voxel is
and, implicitly, how *desirable*, because fast voxels arrive early and get
bought early.

Those are different quantities, and conflating them is the `wm_gain` problem in
general form.

The alternative carries two fields. `T(x)`, arrival time, is migration physics.
`V(x)`, value, is the probability tumor ends up there. Spending the budget
becomes: maximize expected tumor covered subject to a volume limit. Every voxel
costs the same 0.001 mL, so the optimum is still just a sort — rank by `V`, take
the top N. Only the ranking key changes.

It also gives the non-physical terms an honest home. Functional sparing is
currently a speed penalty, which literally claims tumor migrates slower through
the motor strip. That is false. As a bid discount it says "we would rather not
spend budget here", which is what we actually mean.

**Why not now.** `V` has to be fitted, and fitting it at n=5 would be fitting
noise. Revisit when the full cohort lands.

---

## Two things the clinical research changed

### Our baseline is the 46 Gy volume, not the 60 Gy volume

`MARGIN_MM = 20` grown from the whole segmentation — necrosis, enhancement and
FLAIR — is **RTOG/NRG CTV_4600**, which receives 46 Gy `[nrg007]`. The 60 Gy
volume grows the same 2 cm from enhancement and cavity only, with edema left
out, and is roughly **1.8x smaller** `[ibis25]`.

So "2 cm" is right, and it is the margin for both phases. What it does not say
is which GTV it is measured from, and we picked the larger one.

**Why this matters.** A claim of the form "our target beats the standard 60 Gy
plan" would be comparing against a volume nearly twice the size of the one named.
That is large enough to invalidate a headline on its own.

**What to do.** Compute both. `CTV_4600` = 2 cm from NETC + ET + SNFH, which is
what we already have; `CTV_6000` = 2 cm from NETC + ET only. It is a one-line
mask change and it makes the question moot.

### Every target so far is built on a pre-operative scan

Timepoint `_11` in UPenn-GBM is the **pre-operative** study. Confirmed from the
data: `RC` is 0.0 mL on all thirteen subjects processed, because there is no
resection cavity to segment.

Every guideline above defines the GTV on the **post-operative** MRI, from the
resection cavity plus residual enhancement, and each spends paragraphs on
telling cavity change and post-surgical infarct apart from residual tumor.

**Why this matters more than the margin question.** We are drawing a
post-operative target on pre-operative anatomy. The tumor is intact and there is
no cavity, so the seed is a different object from the one every protocol names.

**What to do.** Either reframe the study as predicting spread from the
pre-operative tumor, which is honest and is what the two untreated pairs
actually support, or move to the post-operative timepoint — of which UPenn-GBM
has only 60 across the whole collection.

This has to be settled before anything is written up. It is a framing decision,
not a bug.

### The reach cap is a distance, not a travel time

`ceiling_mm` used to threshold the arrival-time field. That was wrong twice
over.

Travel time is distance over speed, and the field is normalized by its own peak,
so **raising any gain lowered every ordinary speed relative to that peak** and
quietly shrank how much territory the cap allowed. The two arms of every
comparison therefore had different reach: barriers-only could get to 270 mL
inside "20 mm" while the connectome configuration could get to 155 mL. And the
recurrence literature the 20 mm comes from measures millimeters with a tape,
not travel time, so the citation never licensed what the parameter did.

It now caps on Euclidean distance and keeps travel time for the ordering:

```
allowed = within ceiling_mm of the tumor   AND   arrival time is finite
ranked  = by arrival time
```

Distance bounds the reach and does not move when a parameter moves. A finite
arrival time keeps the barriers honest, so nothing behind the falx is bought
just for being close. Travel time still decides what gets bought first, which
is the whole method.

**What it fixed.** Reachable volume on subject 55 is now 327.5 mL for every
parameter setting tested, from all gains at 1 to all at maximum — it was moving
between 74 and 270 mL before. The share of the budget the speed field actually
decides went from 9-39% to 65-70%. Every experiment before this was steering
between a tenth and a third of the target and calling the result flat.

**What it cost.** The narrow band had to go, since it was also measured in
travel time. A solve is slower without it.

**What would change it.** Nothing obvious. If a future version wants the cap to
mean "reachable within 20 mm of travel along the fastest path", that is a
different and defensible quantity, but it needs its own citation and it must not
be described as a distance.

### The corridor null finally separates, and the streamlines win

Four attempts. The first three were flat for three different reasons, none of
them the method's:

| attempt | why it could not detect anything |
|---|---|
| tube 6 mm | real and random corridors overlapped 81%, so both arms got the same field |
| core-bound | tumor plus the 5 mm floor ate 60-90% of the budget |
| budget above reach | the target took everything reachable, so nothing was chosen |

The condition for this experiment to mean anything is that the target must
**reject** volume: `reachable - budget > 0`, so the speed field is forced to
choose. At the full 2 cm budget that quantity was negative on every subject.

At a budget that rejects 45% of the reachable region, with a 2 mm tube:

| subject | budget | rejects | real | random mean | sd | z | beat |
|---|---|---|---|---|---|---|---|
| 00055 | 180 mL | 147 mL | 0.9168 | 0.8980 | 0.0048 | 3.9 | 6/6 |
| 00285 | 189 mL | 155 mL | 0.7430 | 0.7372 | 0.0024 | 2.4 | 6/6 |
| 00352 | 196 mL | 160 mL | 0.5005 | 0.4842 | 0.0036 | 4.6 | 6/6 |

The streamlines that actually run through the tumor beat every random draw on
every subject, by 0.6 to 1.9 points of coverage.

**What this licenses.** Six draws puts the rank-based p at 1/7 for one subject;
three independent subjects each at that rank is about 0.003 combined. That is a
real signal, not a proof. A publishable permutation test needs on the order of a
hundred draws per subject, which is roughly an hour of compute and has not been
run.

**What it does not license.** Nothing about clinical benefit, and nothing at
n=3. It says the tract geometry carries information about where recurrence
lands, measured at one budget and one tube width.

**What would change it.** A hundred draws that put the real value inside the
null distribution rather than beyond it. Or the effect vanishing once the full
cohort lands.

### A hundred draws, five subjects: one of them goes the other way

Same test, 100 draws per subject, tube 2 mm, budget rejecting 45% of reachable,
all parameters at defaults.

| subject | real | null mean | sd | gap | z | beat | p |
|---|---|---|---|---|---|---|---|
| 00055 | 0.9040 | 0.8857 | 0.0054 | +0.0183 | 3.4 | 100/100 | 0.010 |
| 00285 | 0.7365 | 0.7287 | 0.0026 | +0.0078 | 3.0 | 100/100 | 0.010 |
| 00352 | 0.5018 | 0.4860 | 0.0040 | +0.0158 | 4.0 | 100/100 | 0.010 |
| 00129 | 0.5618 | 0.5713 | 0.0021 | -0.0095 | -4.5 | 0/100 | 1.000 |
| 00307 | 0.8302 | 0.8165 | 0.0101 | +0.0137 | 1.4 | 90/100 | 0.109 |

p = 0.010 is the floor for 100 draws, so three subjects are pinned at it.

**00129 is a real reversal, not noise.** Its true corridors were beaten by every
random draw. It is the subject already recorded above as recurring into cortex
while the target spent its budget along white matter: the corridors describe
where cells travel, and on this subject that is not where they settled.

**The combined p depends entirely on which question is asked.**

- Fisher's method gives chi2 = 32.1, df 10, **p = 3.8e-4**. It cannot register a
  reversal: a subject at p = 1.000 contributes zero, so going significantly
  backwards costs the statistic nothing. It answers "is at least one subject
  non-null", which is not the claim.
- The claim is about patients, so the patient is the unit. On the five gaps:
  mean +0.0092, **t = 1.85, one-sided p = 0.069**; Wilcoxon **p = 0.094**; sign
  test 4 of 5 positive, **p = 0.19**.

**What it licenses.** Within most subjects the tract geometry carries
information about where recurrence lands, overwhelmingly so. Across subjects,
at n = 5, there is no population-level effect, and one subject shows the
information can point the wrong way.

**Declared now, before the cohort is run:** the patient-level test on the gaps is
the primary statistic for this null. Fisher is reported, labeled for what it is.

**What would change it.** The full cohort. If reversals like 00129 turn out to be
the recurrence-into-cortex subjects specifically, that is a finding about routes
versus destinations and argues for the disruption term, not against the corridors.

## Endpoints, declared before the numbers exist

Written down now, deliberately, so that neither can be chosen later on the
strength of how it came out.

**Primary: coverage of the whole later tumor**, at matched volume, against a
uniform expansion of the same volume. This is the conservative choice. Its
denominator is dominated by the region inside `tumor + floor_mm`, which every
method includes unconditionally and none of them chooses, so differences are
compressed roughly threefold. That compression costs power; it cannot
manufacture a difference, because the same region sits inside the comparator
too.

**Secondary: coverage of recurrence outside `tumor + floor_mm`.** Defined
structurally, never by looking at where the arms happen to differ. This is the
part any method actually predicts, and it is where the real effect size lives -
31 to 72% of the recurrence sits inside the core depending on subject, so the
secondary is measured over the remaining 28 to 69%.

**Why not make the secondary primary.** It was identified after seeing that the
interesting variance lives there. Promoting it now would be choosing the metric
that flatters the method. Reporting both, with this note, is honest; swapping
them is not.

## The endpoint changed: less brain at the coverage the standard plan achieves

**Supersedes the primary above, on 2026-09-27, at the project lead's direction.**

The project's own goal was always stated as *at any given volume, favorably less,
beat the isotropic version*, which is volume at equal coverage. The section above
made coverage at equal volume primary instead, and that was the wrong way round
for this data.

What forced it into the open was not a method result. `measure.overlap` showed
the 2 cm plan already contains essentially all of the later **enhancing** tumor
in most patients (median 99% within 2 cm of the core). At the plan's volume,
coverage of that compartment has nowhere to go, so a coverage-at-equal-volume
test there cannot show anything either way. The question that remains open is
how little brain it takes to keep that coverage.

**The endpoint, per patient** (`planeval.spared`):

1. The coverage to match is whatever the RTOG 2 cm plan catches of that
   patient's later tumor.
2. Find the smallest uniform expansion that still reaches it, and the smallest
   connectome target that still reaches it, each to within half a percentage
   point, by bisection on volume.
3. **Headline: the target's volume minus the uniform expansion's**, in mL. The
   saving against the plan itself is reported too, but most of it is earned by
   shrinking the uniform margin, which needs no model at all; the method is only
   credited with what it saves beyond that.
4. A target that cannot reach the coverage at all, because a barrier or the
   20 mm reach sits between the tumor and part of the recurrence, is reported as
   a failure and counted as a loss in the sign test, never silently dropped.

Across patients the test is paired and at the patient level, Wilcoxon on the
per-patient savings, for the reason recorded under the hundred-draw null.

**Which compartment is still undecided** (see the section on that); both the
whole segmentation and enhancing tumor are computed.

**Honesty about timing.** One patient, 00055, was computed as a smoke test before
this was written (ET: uniform 165 mL, target 150 mL; whole: 305 vs 281 mL). No
other result on this endpoint had been seen. The old primary is kept and
reported as secondary; the corridor null remains valid for what it measured,
coverage of the whole segmentation at matched volume.

A useful side effect: this endpoint satisfies the precondition that sank the
first three null runs automatically. It asks for the least volume, so the target
is always forced to reject most of what it could reach.

### Three faults the first run of the new endpoint exposed

The first pass over 21 patients was discarded. Each fault below favored one arm
or corrupted a patient, and all three were fixed before the rerun was looked at.

**The uniform comparator had no floor.** Our target can never go below tumor
plus 5 mm. The smallest uniform expansion could, all the way to 0 mm. Where the
later enhancing tumor sits inside the first outline, which is common, a 0 mm
"expansion" matched it and ours lost by exactly its floor: 00052 went 102 mL
against 177. No one treats with a 0 mm margin, so the uniform expansion is now
held to the same 5 mm minimum. Its floor ignores the walls and ours respects
them, so it is a few voxels larger than ours; that leans slightly against us.

**A patient with nothing to cover was scored.** 00160 has no enhancing tumor at
the later scan. Coverage of an empty set is 0, so every shape "matched" at its
minimum and ours was charged 70 mL. A compartment absent at the later scan is
now skipped for that patient and reported as skipped.

**The walls were unreachable even from their own side.** Four patients failed
outright because the target could not reach the plan's coverage. For three of
them, 89 to 100% of the unreachable recurrence lay on wall voxels themselves:
the falx and tentorium are drawn about two voxels thick, they run along midline
tumors, and in 00129 over 3,000 wall voxels fall inside the first tumor. Speed
zero there means the front never arrives, so a recurrence lying against the
falx could not be covered, when a plan clipped at the falx would still dose it.

`target.risk` now gives a wall voxel an arrival time one step past its earliest
open neighbor, after the march. That makes the near face of a wall reachable
and nothing beyond it, because nothing is propagated through: the march is not
rerun. Unreachable enhancing recurrence fell from 10.3% to 1.5% (00129), 6.3%
to 0.3% (00352) and 4.2% to 2.2% (00122). Carving the wall out of the tumor was
considered and rejected: it opens a hole in the falx exactly where the tumor
touches it and lets the front into the other hemisphere.

What remains unreachable is real: tissue sealed behind a wall, or recurrence
on the far face of one. 00307's small shortfall is different in kind, about 1%
lost to edge smoothing at the reach limit, which the unsmoothed plan does not
pay. Every earlier result, the corridor nulls included, was computed under the
old wall rule; the change touches every target equally, but those numbers are
from before it.

**Also:** an analysis that asked for a patient with no tissue map quietly ran
SynthSeg to make one, and did so while the cohort job was running SynthSeg,
which is how 00036 was killed once for memory. `measure.pairs(tissue=True)`
now skips such patients instead.

### First result on 27 patients: the best case, after the three fixes

Default dials, nothing tuned. This is the best-case reading, which picks each
volume from the later scan, so under the three-level plan it is supporting
evidence only.

| recurrence counted as | patients | ours smaller than the best uniform expansion | median saved | test |
|---|---|---|---|---|
| whole abnormality | 27 | 23 of 27 | 27 mL (8%) | Wilcoxon p = 0.001, sign p < 0.001 |
| enhancing tumor | 25 matched, 1 unreachable (00122), 1 skipped (00160) | 16 of 25 | 3 mL (1%) | Wilcoxon p = 0.34, sign p = 0.16 |

On the whole abnormality the shape clearly helps. On enhancing tumor it is a
tie, with a few large losses (00307, 00353 and 00183 each need 160 to 200 mL
more than the uniform expansion) that have not been examined yet. So which
compartment counts as the recurrence is no longer only a definitional question:
at the defaults, it decides whether there is a result.

## What the structure-versus-optimizer argument settled

Two advocates argued it out. They converged on four things, which are now done:

- **`risk_gain` was on an incomparable scale.** The at_risk share never exceeds
  about 0.25, so `risk_gain = 1` was a x1.035 nudge while `wm_gain = 5` was a
  flat x5. The optimizer's box ran 0 to 4, so the term could not have mattered
  at any point in the search. The earlier conclusion that disruption contributes
  nothing was produced by the parameterization, not discovered. Now divided by a
  **fixed** 0.25, so `risk_gain = 4` means x5 at the most disrupted parcel. Fixed
  rather than per-subject on purpose: a cohort-derived scale would make the
  parameter mean something different every time a subject is added.
- **`sigma_mm` was twice the physics.** A gaussian's 80-20 edge is about 1.7
  sigma, so 5.0 mm produced an 8.4 mm penumbra against a real linac's 3-5 mm.
  Now 2.5, giving 4.2 mm. This is a machine constant and is not for fitting.
- **The smoothing stays.** The corridor null separated at z 3.9 to 4.6 with it
  applied, so it demonstrably did not erase the one real effect there is.
- **The mandatory core now respects reachability.** It is Euclidean and knew
  nothing, so it bought tissue the speed field had priced at zero, including up
  to 5.7 mL across a barrier. Microscopic disease abutting the margin does not
  extend to the far side of the falx.

**And one thing they settled against changing.** `floor_mm` stays a hard 5 mm
rather than becoming a fitted parameter, because it has **no interior optimum**:
free budget always improves coverage, so an optimizer would drive it to zero and
report that as a finding. The same argument forbids fitting `sigma_mm`, where
zero is physically impossible. Parameters whose optimum sits on a boundary
cannot be fitted; you only recover the boundary. The floor gets a **sensitivity
sweep at 0 / 2 / 5 / 8 mm** instead, reported with everything else held.

## Registration between timepoints is not a problem

Checked, because a target and the recurrence it is scored against reach MNI
through two separate warps. Brain dice 0.987 to 0.989 across five subjects, mean
residual deformation 0.92 to 1.38 mm, 95th percentile 3.2 to 4.1 mm. The mean is
well inside a voxel. The tail is comparable to the effects being measured, so it
is worth stating as a limitation, but it is not corrupting anything.

## The fit's budget is a share of reachable, not of the clinical plan

`fit.BUDGET` was **0.4 of the subject's RTOG plan**, while the corridor null used
a budget **rejecting 45% of the reachable region**. Since the score is only
sensitive to the parameters when the target has to reject volume, the fraction
rejected is the thing that has to be held fixed, and two different denominators
cannot both hold it fixed.

**The size of the problem was overestimated when it was first raised.** The claim
was that plans running 265 to 926 mL against a roughly constant reach would make
the rejected fraction swing wildly. That reasoning was wrong: the reach is not
constant. `rtog` is a 20 mm shell around the segmentation clipped to the brain,
and the reachable region is the same shell clipped to the brain *and* to what is
not behind a barrier. They are the same object minus the dural block, so they
track each other closely. Measured across a five-fold range of tumor size:

| subject | reach | plan | reach/plan | rejected at 0.4 x plan |
|---|---|---|---|---|
| 00133_11 | 164.9 | 165.0 | 0.999 | 0.600 |
| 00140_11 | 163.2 | 170.6 | 0.957 | 0.582 |
| 00088_21 | 814.6 | 872.0 | 0.934 | 0.572 |
| 00093_21 | 854.1 | 926.5 | 0.921 | 0.566 |

So the real spread was about three points, not the large one asserted. The only
thing that varies is how much of each subject's shell sits behind the falx or
tentorium.

**Changed anyway**, to `0.55 * target.reachable`, because it costs nothing and
removes the variation by construction rather than by luck: every subject now
rejects exactly 45%, the same question the corridor null asks. What varies
instead is the budget as a share of the clinical plan, which is a reporting
quantity and belongs in the output. `fit.rejected(pairs)` prints both.

`target.reachable` is cached per subject and is parameter-independent: every
gain scales a strictly positive speed, so which voxels end up with a finite
arrival time is fixed by the barriers alone.

**What would change it.** Nothing about the denominator. The remaining question
is whether 45% is the right amount to reject, which is an operating point rather
than a definition, and the coverage-versus-volume curve is what answers it.

## Which compartment counts as the recurrence

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

## Open questions with no owner

- **What is the correct `wm_gain`, if the concept survives at all?** The sweep so
  far says the median prefers 1.0, i.e. no tissue preference — but subject 00352,
  the one with heavy callosal involvement, prefers 5.0 and gains ten points by
  it.

  An earlier draft of this entry called that heterogeneity and inferred a
  patient-dependent parameter. **That was stated past the evidence.** One subject
  in a handful preferring a different value is not distinguishable from noise,
  and the sweep moves one parameter at a time, so it cannot see the interaction
  either. The 40-subject cohort is what settles it.

  The more economical reading, which costs no new parameter: `wm_gain` speeds up
  **all** white matter identically, everywhere, in every subject. `tract_gain`
  speeds up only the bundles that pass through *this* tumor. A tumor whose escape
  route happens to be a white matter highway is helped by either; every other
  subject is only inflated by the global one. So the median preferring 1.0 may be
  the data saying *steer with the corridors, not with the tissue class* — which is
  what the corridor null found from the other direction.

  **This does not argue for more tissue multipliers.** A dial for a tissue type
  only a few subjects' tumors touch is fitted on a few subjects. The test for any
  new parameter is whether it improves the **held-out** score, never the fitted
  one. The genuinely unmodelled thing is direction — white matter is anisotropic,
  and we approximate that with thin corridors rather than tensors.

- **Does the connectome earn its place?** Partly answered. The tract corridors
  carry information within subjects: at 100 draws, four of five subjects beat
  the null and three at the floor. But one reversed at z -4.5, and across
  patients the effect is p = 0.069 at n = 5. See the hundred-draw section above. The disruption term does not: permuting
  the connectivity matrix changes nothing, because at `risk_gain = 1` it is a
  3.5% speed nudge beside a 400% one. Either give it real weight and retest, or
  drop it and say so.

- **Which RTOG volume is the comparator, the 46 Gy or the 60 Gy one?** Settled
  in part above: we currently compute the 46 Gy one. ASTRO 2025 `[astro25]`,
  which our own supervisor co-authored, gives the range as **10-20 mm** rather
  than a fixed 2 cm and calls both the one-phase and two-phase approaches
  acceptable, so "2 cm is the US standard" is too strong a claim to make.

- **Settled: the margin.** Dr. Knisely confirmed 2 cm from the edge of the
  entire segmented tumor, which is what `margin_ml` computes. The 46/60 Gy
  distinction only ever named which starting volume a protocol meant; we are
  measuring volumes, not prescribing dose, so it changes nothing in the code. In
  writing, call the baseline "2 cm from the whole segmentation, per RTOG/NRG
  CTV_4600" rather than naming a dose.

- **Settled: pre-operative scans are fine.** The goal is to beat a uniform
  expansion at covering recurrence for a given volume. Both arms grow from the
  same seed, so the comparison holds on pre-operative anatomy. What it does not
  support is calling the output a clinical CTV ready for planning. Note that the
  earlier argument for this - "RC is 0 mL everywhere so these must be pre-op" -
  was wrong: RC is 0 on the post-operative scans too, because the segmenter
  never emits that label at all. The claim rests instead on the UPenn-GBM paper
  reporting 611 pre-operative and 60 post-operative scans, and on a craniotomy
  visible in subject 00130's second timepoint.

- **Is ESTRO 15 mm or RTOG 2 cm the comparator a reviewer expects?** They are
  different protocols measured from different things — ESTRO from enhancement
  plus cavity with edema excluded, RTOG phase 1 from the FLAIR abnormality. Our
  `margin_ml` grows from the whole segmentation including edema, so it is closest
  to RTOG phase 1, which is the 46 Gy volume rather than the 60 Gy one. Report
  both, labelled, and lead with whichever is smaller.

- **Does the SVZ rule fire correctly on a contacting subject other than 55?**
  It returns `(False, True)` for 55 and `(False, False)` for 307, which is
  correct for both. That is one positive case. It has never been checked on a
  tumor touching the left rind, or on one abutting the temporal horn.

- **Does SynthSeg hold up on a tumor sitting deep in white matter?** It
  hallucinates inside the lesion on 55, which is harmless there because the
  tumor is the seed. On a deep tumor the fabricated tissue could extend into the
  rim the front actually crosses. Unchecked.
