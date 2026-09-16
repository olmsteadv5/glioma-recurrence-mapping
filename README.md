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

The stages run in order and each one needs the one before it, so you restart at
a stage rather than picking stages out of the middle:

```bash
./main.py 36 --from register
```

| `--from`   | what runs                        | roughly |
|------------|----------------------------------|---------|
| `fetch`    | everything, the default          | 3 min   |
| `strip`    | SynthStrip onwards               | 2 min   |
| `segment`  | BraTS onwards                    | 2 min   |
| `register` | greedy and the atlas work        | 40 s    |
| `map`      | parcel overlap and function only | 3 s     |

`--from map` is the one to reach for after changing anything in `atlas.py` or
`function.py`, since it reuses the segmentation and the warp.

It says what is missing if you start too late, rather than failing part way
through. `--no-view` skips the browser window at the end.

`--timepoint 2` fetches the follow-up scan instead of the pre-operative one.
Not every subject has one; 36, 42 and 51 do, 2 does not. Run both and a brace
in the subject list ties them together, with the change in WT between them.

To look at a subject you have already done, don't rerun anything, just open the
view:

```bash
./view.py 36
```

Every parcel the tumour touches gets drawn. The legend is the main control:
tumour tissue classes first, then the tracts, then the brain surface, then the
parcels in order of involvement, so the things you care about are not buried.
Click any entry to toggle it.

Along the bottom are the things the legend cannot do: a slider for brain
surface opacity, a dropdown to show or hide all the parcels at once, and a dropdown to
switch the parcel coloring.

Two palettes. The atlas's own colors parcels by region, so neighbours often
look identical, only 75 distinct colors across 696 parcels. The other is
YBA_696_Freeview_HighContrast.txt, where no two touching parcels share a
color, which is much easier to read when a tumour spans a dozen of them.

From Python, `view.run` also takes `top=20` if you only want the most involved
parcels, `shell_opacity` for the brain surface's transparency, and `palette`
to pick which coloring it opens with.

## Terminal UI

```bash
./tui.py
```

Four panels: the inputs down the left, a viewer, a breakdown on the right, and
the pipeline along the bottom.

The pipeline panel is a prompt. Press `t` or click it, type a command, press
enter. Up and down walk the history, escape gives the keyboard back. It only
knows about this project, it is not a shell:

`save` at the prompt writes whatever the viewer is showing next to the outputs:
the 3D at 1600x1200, or the three slice planes side by side at 2100x700, both
larger than the panes they came from.

```
./main.py 36 42    the whole pipeline for those subjects
--timepoint 2      on any of these, the follow-up scan
fetch 36           \
strip 36            |  restart at that stage and run everything after it,
segment 36          |  same as ./main.py 36 --from <stage>
register 36         |
map 36             /
show 36            load a subject into the viewer
browser 36         that subject in a separate browser window
cancel             stop the job, and the segmenter's container with it
clear              clear the log
help               list the commands
```

`map 36` is the quick one, a few seconds, and it is what you want after
changing the atlas or function code. It runs with `--no-view`, since the viewer
is already sitting next to the prompt.

The viewer is either the 3D atlas, where dragging rotates and the scroll wheel
zooms, or an ITK-SNAP style 2x2 of axial, coronal and sagittal with a cursor
readout. Clicking or dragging in any plane moves the crosshair in all three.

The modality strip sits in the viewer's bottom border, away from the other
controls on the right. Press `1` to `4` or click it. The number of the one in
use is bright, and anything the current space does not carry is greyed.

The cursor panel gives every sequence's intensity at that voxel, not just the
one on screen, and in MNI it names the parcel under the crosshair.

MNI also draws the involved parcels onto the slices, in the same colors the
3D uses, under the tumour rather than over it. The selected one is brighter.
That is why `p` works there and not in native space: native has no parcels to
color, so the title says which palette is on only where it means something.

```
v  3d / slices          1-4  modality           t  focus the prompt
s  native / mni         a b c  breakdown view    x  cancel the run
p  parcel coloring                              q  quit
```

Half of those only make sense in one of the two viewer modes. The ones that do
not apply go grey in the footer rather than doing nothing silently.

Each panel carries a strip along its bottom left of the things it can show —
the four modalities in the viewer, the three breakdown views — with the key for
each one and the current one bright. The rest of the controls sit bottom right,
labelled with what they give you rather than with what you are looking at, so
in native space `s` reads `s mni`.

Each parcel row carries a chip in that parcel's own color, so a shape you can
see in the 3D can be found in the list without clicking through it. The chips
follow the palette toggle.

The layer checkboxes drive both views: unticking `SNFH oedema` takes it out of
the 3D and out of the slice overlay, so there is no separate overlay key.

`tumour` above them is the three classes contoured as one closed surface rather
than three that meet along internal faces, with its parts indented under it. It
is a real parent: unticking it takes the parts with it, and it draws its own
combined shape only when none of the parts is on, so the two can never wash
each other out.

For a patient with two timepoints, a rule at the bottom of the layers divides
off three more: what both scans have, what is new in the later one, and what
has gone since the first. They are computed in MNI, the only frame the two
share, and they are the shapes behind the Dice in `changes`.

Every layer draws in both views. The margin is an outline on the slices and the
rest are fills, so anything you tick shows up whichever viewer you are in.

`margin 2 cm` above that is the whole tumour grown by 20 mm, the usual CTV
expansion, **clipped to the brain**: a fifth to a quarter of a raw expansion
lands in skull and air, which would put subject 36 at 751 mL instead of 549.
It is an outline on the slices rather than a fill, since 2 cm of fill hides the
thing you grew it from. Both the surface and the slice mask are built the first
time you tick it, because the distance transform is not free.

The volume is measured in native space, where the millimetres are the
patient's, not in the warped MNI copy. Across these six subjects a 2 cm margin
is 3.3 to 7.5 times the tumour it came from, the multiplier being largest for
the smallest tumours.

The breakdown lists the layers with their volumes, clickable to show and hide,
then one of three views: the involved parcels, the functions at risk, or the
connectivity at risk. Whatever is selected
lights up in the 3D and every other parcel drops back, so a row and a shape in
the viewer are the same thing. Every list opens with the same two rows, which
are the only control over how much of the atlas is drawn:

```
  no parcels            nothing, so the tumour is clear
  all parcels       99  everything, nothing singled out
  L_S_F            94%  that one bright, the rest faint
```

The second one also carries a note on what the list is and how it was built.
The parcels are not a layer checkbox, because unlike the tumour classes or the
brain surface they are a collection with a list attached, and two controls over
the same thing disagree at the ends: `all parcels` and an unticked box cannot
both win.

In the slice view, picking a parcel puts the crosshair inside it instead. Only
MNI knows where the parcels are, so it switches space if it has to. It works both ways: click a parcel in the 3D and
its row is selected, drag and it rotates instead.

WT, TC and ET are unions of the tissue classes, not classes themselves, which
is why none of their volumes matches a single layer. The inputs panel spells
out what each one contains and leaves them uncolored, since the three colors
belong to the classes in the breakdown.

Under them the panel carries `where` the tumour sits, by hemisphere and lobe,
which is the same question the 119 parcel rows answer one at a time, and `run`,
which segmenter produced this and on what, how long it took, its Dice where
there are reference labels, and `aligned on`: the share of the brain the
registration had left to work with once the lesion was masked out of the metric.
That last one is about a single registration, not about change over time.

Click a subject or focus the panel and arrow through them; it scrolls to keep
the chosen one in view, and steps over any whose outputs have been deleted.

The panel is in two parts. `stats` holds the tumour, where it sits and how it
was made; the lobe volumes are indented under `in parcels` because that is the
volume they add up to, and they sit under a hemisphere only when the tumour is
in both, since otherwise the hemisphere is the whole of it.

There are four tissue classes but only three of them usually appear. `RC`, the
resection cavity, is a post-treatment class, so its region row and its layer
show up only on a subject that has one.

Between them, `changes` appears when a patient has more than one timepoint: the
change in WT, how many parcels the tumour newly reaches or has left, the Dice
between the two tumours, and how much of the later one sits inside the earlier
one's footprint. Subject 36 comes back at Dice 0.56 with 91% of the recurrence
inside where the original was.

Those two are not the same question. Dice is low because the tumour shrank, not
because it moved; containment says it came back in the same place.

`checks` holds `stages` and `atlas`, a row each. The strip row also carries
whatever `verify.py` objected to, which used to reach you only as a traceback. A tick is quiet and carries
how long the stage took and how long ago it ran. A cross is the only colored
thing in the panel, and it opens a red line saying what is actually wrong:
which modality never downloaded or never stripped, or that a partial rerun left
this stage behind the one before it.

While the pipeline runs, its panel shows which stage is going and the viewer
switches to the subject being worked on as soon as its scans land, so you are
looking at the case you are waiting for rather than the last one.

For a parcel the panel underneath gives its anatomy and its own Parcelsynth
terms, and for a function every parcel that produced the score, with how much
of it the tumour occupies and what that contributed; those shares add up to the
score. Both lists are complete and the panel scrolls.

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

Almost every term comes out above zero, so the list is cut at a score of 0.01
rather than at a rank. That leaves 170 to 230 terms, which is a real tail, not
a top ten. Reading a term gives the parcels that produced its score and what
each contributed; those shares add up to the score.

```
UPENN-GBM-00036_11   movement, sensorimotor, finger, hand   left sensorimotor
UPENN-GBM-00002_11   pain, noxious, auditory, listen        right amygdala, insula
```

These are **meta-analytic associations pooled across published studies**, not
this patient's functional anatomy. They suggest what to look into. They do not
replace fMRI mapping or awake craniotomy, and they must not be used to set a
margin.

## Connectivity at risk

`connect.py` reads the Yale structural connectivity table. It is an edge list,
not a matrix: 241,860 rows, one per parcel pair, with sixteen metrics each. It
folds `SC_Count` into a 696x696 array, cached, and then for each parcel the
tumour has **not** touched, sums the connections it has into parcels the tumour
**has**, weighted by how much of each of those is gone.

That answers a question the overlap cannot: which intact regions are losing
their input. Selecting one lights it in the 3D and dims everything except the
involved parcels it connects to, and the panel underneath lists those with how
much of the loss each accounts for.

```
L_SM2_B   Left supramarginal gyrus 2 B, parietal, not in the tumour
          25% of its connectivity at risk
          L_SM1_D  52% involved, 18.8% of the loss
          L_SM1_E  71% involved, 18.7% of the loss
```

This is a **healthy population mean**, the same caveat as Parcelsynth. A tumour
displaces tracts, so this patient's real connectivity differs. It says where to
look, not what has happened. That note, and the equivalent for the other two
views, is on the first row of each list.

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

The faint grey brain is one closed surface contoured out of MNI152, not the 696
parcel meshes. Those are 754 separate closed bodies, so stacking them as a
translucent shell piles up alpha along every view ray until the involved
parcels inside end up darker than the context around them.

If you have an NVIDIA GPU it uses BraTS25_1, which won the 2025 challenge. On
CPU it falls back to BraTS25_2 and says so. Nothing to configure either way.

main.py has a shebang pointing at .venv, so ./main.py works without activating
anything. Fix that line if you move the project.
