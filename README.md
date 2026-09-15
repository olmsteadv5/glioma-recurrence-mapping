# Brain tumour segmentation on UPenn-GBM

Segments glioma MRI with the BraTS competition algorithms, then works out which
Yale Brain Atlas parcels the tumour sits in and which white matter tracts run
through it. Opens a 3D view at the end.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Docker has to be running and you need to be in the docker group. FreeSurfer and
ITK-SNAP also need to be installed, for mri_synthstrip and greedy.

## Running it

```bash
./main.py 36 42 51
```

The numbers are UPenn-GBM subject numbers, anything from 1 to 630. It downloads
the scans, skull-strips them, segments, registers into atlas space, works out
the parcel overlap, and opens a browser view for each one.

About 3 minutes a subject on CPU. The first run is a lot slower because it
pulls a 30 GB Docker image and 60 MB of atlas data, both of which are kept.

Everything else reruns every time, so if you only want to look at a subject you
have already done, don't rerun the pipeline. Just open the view:

```bash
./view.py 36
```

Every parcel the tumour touches gets drawn. The legend is the main control:
tumour tissue classes first, then the tracts, then the grey shell, then the
parcels in order of involvement, so the things you care about are not buried.
Click any entry to toggle it.

Along the bottom are the things the legend cannot do: a slider for shell
opacity, a dropdown to show or hide all the parcels at once, and a dropdown to
switch the parcel colouring.

Two palettes. The atlas's own colours parcels by region, so neighbours often
look identical, only 75 distinct colours across 696 parcels. The other is
YBA_696_Freeview_HighContrast.txt, where no two touching parcels share a
colour, which is much easier to read when a tumour spans a dozen of them.

From Python, `view.run` also takes `top=20` if you only want the most involved
parcels, `shell_opacity` for the shell's starting transparency, and `palette`
to pick which colouring it opens with.

## Terminal UI

```bash
./tui.py
```

Four panels: subjects, a viewer, the pipeline log and a breakdown. `r` runs the
pipeline on the selected subject and streams the log, `c` cancels. `v` switches
the viewer between the 3D atlas and orthogonal slices, `s` switches slices
between native and MNI, `1`-`4` pick the modality, `p` swaps the parcel
colouring, `f` swaps the breakdown between parcels and functions, arrows rotate
or scrub.

**kitty only.** It draws images with kitty's graphics protocol, which no other
terminal here supports. Elsewhere it still runs but says the images will look
poor. The browser viewer is the portable one.

Slices default to **native** space, because that has all four modalities and is
undistorted. MNI has only T1 and is a warped brain, but it is the only space
where atlas coordinates mean anything, so the readout shows MNI mm there and
says "no anatomical origin" in native.

## Functions at risk

`function.py` reads Parcelsynth, which scores all 696 parcels against 334
function terms. For a tumour it weights each involved parcel by how much of it
is destroyed, then reports terms **over-represented against the whole-brain
average** rather than raw scores, so common terms do not win every time.

```
UPENN-GBM-00036_11   grasp, arm, tactile, touch, finger    left sensorimotor
UPENN-GBM-00002_11   unpleasant, aversive, anxiety         right amygdala, insula
UPENN-GBM-00042_11   motion, auditory, sound               left temporal
```

These are **meta-analytic associations pooled across published studies**, not
this patient's functional anatomy. They suggest what to look into. They do not
replace fMRI mapping or awake craniotomy, and they must not be used to set a
margin.

## Where things end up

```
inputs/<SUBJECT>/     the raw scans, never touched after download
inputs/atlas/         Yale Brain Atlas data
outputs/<SUBJECT>/    stripped scans, segmentation, registration
outputs/results.json  volumes, Dice, parcel overlap, functions
```

Both directories are gitignored. Everything in them can be regenerated.

## Files

```
main.py        runs the whole thing
fetch.py       download scans
skullstrip.py  SynthStrip
verify.py      input checks
segment.py     BraTS in Docker
evaluate.py    volumes and Dice
register.py    SRI24 to MNI152
atlas.py       atlas volume, meshes, parcel overlap
tracts.py      tract centrelines
view.py        3D view, also runnable on its own
paths.py       shared paths
```

## Worth knowing

If you have an NVIDIA GPU it uses BraTS25_1, which won the 2025 challenge. On
CPU it falls back to BraTS25_2 and says so. Nothing to configure either way.

main.py has a shebang pointing at .venv, so ./main.py works without activating
anything. Fix that line if you move the project.
