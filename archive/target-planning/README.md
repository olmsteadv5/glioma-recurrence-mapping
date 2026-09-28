# Target planning, in storage

Put away on 2026-09-28 to work on the baselines at the pace of the lab. Nothing
here is maintained while it sits in storage.

## What it is

A connectome-guided radiation target for glioblastoma: a speed field built from
tissue, dural walls and atlas streamlines through the tumor, an eikonal solve
for arrival time, and a target that takes the earliest-arriving tissue within
20 mm. The goal was the same coverage of the later tumor as the standard 2 cm
plan, with less brain.

- `plan/` the code: `barriers` `conduits` `speed` `march` `target` `baseline`
  `planeval` `fit`, plus a frozen copy of `measure.py` so the folder runs alone
- `DECISIONS.md` every choice, why, and what would change it
- `SOURCES.md` the reference behind every `# why:` tag in the code
- `flowchart/gpsflow.html` the stage-by-stage flowchart
  (https://claude.ai/artifact/L1yeaVBBLQmhsdGBCKRkJc)
- `results/` the 100-draw corridor null and the best-case runs
- `scripts/` the one-off runners behind those results

## Where it stopped

- 100-draw corridor null, 5 patients: real streamlines beat all 100 random
  draws in 3, and lost to all 100 in one (00129). Across patients p = 0.069.
- Best case at equal coverage, 27 patients, nothing tuned: 27 mL (8%) less
  brain than the best uniform expansion on the whole abnormality (p = 0.001),
  a tie on enhancing tumor.
- A review found much of the model does nothing measurable: the walls make
  0.3% of the 2 cm plan unreachable in the median patient, satellites appear
  in 0 of 41, and the ventricle-lining and sparing terms are never switched on.

## What was next

1. An ablation: remove one piece at a time, keep only what the results need.
2. Fix the corridor width, which is effectively twice what it says.
3. The three-level evaluation in the flowchart, with nested cross-validation.

Two questions for the lab decide how it would be scored: whether recurrence
means enhancing tumor or the whole abnormality, and how much coverage a plan
may give up and still count as no worse.

## Getting it back

The exact working state is the tag `target-planning-2026-09-27` and the branch
`geodesic-target`. `git checkout target-planning-2026-09-27` restores it as it
was, with `plan/` in its original place.

To run the copy in this folder instead, from the repository root:

    PYTHONPATH=archive/target-planning:. .venv/bin/python -c "from plan import target"

It imports `paths`, `scan` and `yale` from the root, so if those change the tag
is the version to trust.
