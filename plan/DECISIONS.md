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

## Open questions with no owner

- **What is the correct `wm_gain`, if the concept survives at all?** The sweep so
  far says the median prefers 1.0, i.e. no tissue preference — but subject 00352,
  the one with heavy callosal involvement, prefers 5.0 and gains ten points by
  it. That heterogeneity suggests a patient-dependent parameter rather than a
  global one, which nothing in the design currently supports.

- **Does the connectome earn its place?** Undecided until the shuffled null runs.
  If a degree-preserving rewiring scores the same, the whole connectome branch is
  decoration and the honest paper is about barrier-aware expansion.

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
