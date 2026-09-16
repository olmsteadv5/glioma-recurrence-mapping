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

Four panels: the inputs down the left, a viewer, a breakdown on the right, and
the pipeline along the bottom.

The pipeline panel is a prompt. Press `t` or click it, type a command, press
enter. Up and down walk the history, escape gives the keyboard back. It only
knows about this project, it is not a shell:

```
./main.py 36 42    segment and map those subjects
cancel             stop the running job
browser [n]        open the 3D view in a separate browser window
clear              clear the log
help               list the commands
```

The viewer is either the 3D atlas, where dragging rotates and the scroll wheel
zooms, or an ITK-SNAP style 2x2 of axial, coronal and sagittal with a cursor
readout. Clicking or dragging in any plane moves the crosshair in all three.

```
v  3d / slices          p  parcel colouring     t  focus the prompt
s  native / mni         f  parcels / functions  c  cancel the run
o  segmentation overlay h  reset view           q  quit
1-4  modality           pgup/pgdn  step slice   space  hide a parcel
```

Half of those only make sense in one of the two viewer modes. The ones that do
not apply go grey in the footer rather than doing nothing silently.

The breakdown lists the layers with their volumes, clickable to show and hide,
then either the involved parcels or the functions at risk. **Clicking a row
only selects it**, so you can read the panel underneath; space or enter is what
hides a parcel. Whatever is selected lights up in the 3D and everything else
dims.

For a parcel the panel underneath gives its anatomy and its own Parcelsynth
terms. For a function it gives the parcels that produced the score, each with
how much of it the tumour occupies and what that contributed. Those shares add
up to the score.

**kitty only.** It draws images with kitty's graphics protocol, which no other
terminal here supports. Elsewhere it still runs but says the images will look
poor. The browser viewer is the portable one.

Slices default to **native** space, because that has all four modalities and is
undistorted. MNI has only T1 and is a warped brain, but it is the only space
where atlas coordinates mean anything, so the readout shows MNI mm there and
says "SRI24 grid" in native.

## Functions at risk

`function.py` reads Parcelsynth, which scores all 696 parcels against 334
function terms. For a tumour it sums those profiles, weighting each involved
parcel by how much of it the tumour occupies.

```
UPENN-GBM-00036_11   movement, sensorimotor, finger, hand   left sensorimotor
UPENN-GBM-00002_11   pain, noxious, auditory, listen        right amygdala, insula
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
tui.py         terminal UI
render.py      offscreen 3D and slice images for the TUI
function.py    Parcelsynth terms
paths.py       shared paths
```

## Worth knowing

If you have an NVIDIA GPU it uses BraTS25_1, which won the 2025 challenge. On
CPU it falls back to BraTS25_2 and says so. Nothing to configure either way.

main.py has a shebang pointing at .venv, so ./main.py works without activating
anything. Fix that line if you move the project.
