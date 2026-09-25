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

## Open questions with no owner

- **What is the correct `wm_gain`, if the concept survives at all?** The sweep so
  far says the median prefers 1.0, i.e. no tissue preference — but subject 00352,
  the one with heavy callosal involvement, prefers 5.0 and gains ten points by
  it. That heterogeneity suggests a patient-dependent parameter rather than a
  global one, which nothing in the design currently supports.

- **Does the connectome earn its place?** Partly answered. The tract corridors
  do: real streamlines beat random ones 6/6 on three subjects once the test was
  built so it could detect a difference. The disruption term does not: permuting
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
