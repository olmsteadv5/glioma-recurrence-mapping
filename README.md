# Brain tumor segmentation on UPenn-GBM

Segments glioma MRI with the BraTS competition algorithms, then works out which
Yale Brain Atlas parcels the tumor sits in and which white matter tracts run
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

Every parcel the tumor touches gets drawn. The legend is the main control:
tumor tissue classes first, then the tracts, then the brain surface, then the
parcels in order of involvement, so the things you care about are not buried.
Click any entry to toggle it.

Along the bottom are the things the legend cannot do: a slider for brain
surface opacity, a dropdown to show or hide all the parcels at once, and a dropdown to
switch the parcel coloring.

Two palettes. The atlas's own colors parcels by region, so neighbors often
look identical, only 75 distinct colors across 696 parcels. The other is
YBA_696_Freeview_HighContrast.txt, where no two touching parcels share a
color, which is much easier to read when a tumor spans a dozen of them.

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

`v` cycles the viewer through three: the 3D atlas, where dragging rotates and
the scroll wheel zooms; an ITK-SNAP style 2x2 of axial, coronal and sagittal
with a cursor readout, where clicking or dragging in any plane moves the
crosshair in all three; and the graph.

## The graph

The connectome the tumor sits in. Every parcel is a node, the tumor is one more
in the middle, and there is a line wherever a tract that **runs through the
tumor** ties two of them together. That is the whole rule, and every number in
the view follows from it: irradiate this volume and these are the connections
that go through it.

A parcel is a node if the tumor overlaps it, or if at least three tracts
through the tumor reach it. One or two is a graze and would only crowd a plan.
Subject 2 comes to 233 nodes and 810 lines from its 1034 crossing tracts.

There are two hubs. The yellow one is the tumor. The second, **no parcel**, is
not a place but the absence of one: the far end the atlas has no name for. The
190 tracts that cross the tumor and land in unnamed tissue run to it, and so do
the 173 that miss the tumor but pass through the parcels it has taken, which
would otherwise be nowhere at all on the chart. Because two hubs want the width
and the pane is wider than it is tall, the rim is an ellipse rather than a
circle, with the hubs abreast in the middle. It stops stretching at 2.2 times
as wide as it is tall, so on an ultrawide the extra width goes to the margins
and not into a flatter and flatter shape.

Three kinds of line: **amber** joins two parcels a crossing tract ties together,
**rose** joins a parcel and the tumor, **blue** goes to no parcel. A hub dims like any other node
when the pick does not reach it, and takes a bright ring when it does. One line is
one connection, never one per tract; a thicker line is more tracts along it,
which is what a plan needs to know.

The key names lines and counts the tracts on them, which are different things
and are labeled as such. Its total, `824 lines for 1207 tracts` on subject 2,
matches the left panel's two tract rows added, 1034 through the tumor and 173
clipping a parcel. The rows overlap on purpose: a tract joining two parcels is
one pair line and two spokes, so it is counted on all three. Toggle either
layer and the 3d shows exactly what the chart is built from.

Region colors take their hue from the lobe and their saturation from the
hemisphere, at one lightness held equal in CIELAB. Encoding the hemisphere as
brightness made the right hand regions read as a faded version of the left
rather than as a different side; the same rgb value is a far lighter yellow
than it is a blue, so the equalising has to happen in a perceptual space.

Orange nodes are parcels the tumor overlaps, green ones are parcels it misses,
and the yellow hub is the tumor. A node's **area** is how much of that parcel
is inside the tumor. Clicking a node opens whichever list has a row for that
parcel and selecting a row lights its node. Drag to pan, scroll to zoom, click
the tumor to put it back.

`s` swaps the arrangement, the way it swaps space in the slices. **Spread** is
laid out so the gaps are the real distances between parcels, to a median of
10%. **Circle** puts the parcels on a rim grouped by region and routes every
line through both its regions on the way across, so parallel tracts travel
together and the picture reads as bundles. Circle says nothing about distance
and does not pretend to: no scale bar, no rings, no bow.

Pick a parcel and an inventory of its tracts opens top right; with nothing
picked there is no panel. It heads with what the parcel is, set out
label and value the way the left panel sets its numbers: how much of it is in
the tumor and what share that is, its class and lobe, how many tracts it has,
and how many different parcels they reach. Then one row per tract: the far
end in a column down the left with its parcel named beside it, the tract
running right at its real length in millimeters, the picked parcel on the end
of it, and that length printed alongside. Rows keep a readable height and the
list scrolls on the wheel rather than squeezing; `L_CC_B` has 717 tracts and
they are still one row each. The length is printed alongside, following its own bar rather than
sitting in a column of its own. A grey scale in millimeters runs underneath the
rows rather than over them. Both dots keep the size and color they have on the
chart, and the mass is not named because it is the one yellow dot.

The left hand column is the point of it. For `L_SF8_C` on subject 36 the
longest tracts, 111 to 129 mm, run to `R_SF8_C` and `R_M2_C`, across to the
other hemisphere and into parcels the tumor missed; the middle band at 50 to 61
mm all goes to `L_CC_B`, the corpus callosum, which the tumor overlaps; and the
shortest, 4 to 32 mm, go to the mass itself and to its own neighbours. The
chart cannot show any of this, because a line there is as long as the gap
between the two dots it joins.

Zoom past about twice and every node is labeled; below that only the picked
node and what it connects to. The bar at the bottom is 50 mm, which means
something because the gaps are real distances.

Each region gets a soft ground under its parcels, and a region is a hemisphere
and a lobe, not a lobe. Subject 36 has 55 left frontal parcels and 8 right
frontal ones whose centers land 79 units apart, so one ground for both stretched
across half the picture and named neither. They are `left frontal` and `right
frontal` now, right dimmer than left, and they are named in the key rather than
on the chart.

They still overlap, because the layout is connectivity and not anatomy, and
parcels from different regions genuinely interleave in it.

The key says what every channel means, including the one that is easiest to
misread: three sized dots with the milliliters they stand for, so a bigger dot
is a number rather than an impression. Every row carries its own count, so the
key doubles as the tally for what is on screen.

The nodes are placed by stress majorization, so **the gaps on screen are the
real distances between the parcels**, to a median of 10%. That uses
`SC_EuclDist`, the center to center distance: `SC_Length` would be the mean
streamline length but it is clamped, with over half the edges sitting at
exactly its maximum of 168.8 mm, so it cannot carry lengths.

The tumor's own edges are real geometry: a spoke is drawn to every parcel that
a tract centerline passes through **after passing through the tumor**. For
subject 36 that is 803 centerlines through the mass, 676 of which go on to
reach a parcel, 218 of them into `L_CC_B` alone.

Edges are bundles of arcs, not lines. Where the tract geometry has the pair,
each arc is one real centerline, bowed until the length it is drawn at is the
length that tract actually runs: the piece of it between where it touches one
parcel and where it touches the other. Those are not all the same. The median
is 1.37 times the straight line between the two parcels, a tenth of them are
over 3 times, and two tracts joining the same pair differ in length by about a
fifth. Arcs past 1.8 times are drawn at 1.8, or a handful of long ones sweep
across the whole chart.

Where only the connectivity table has the pair, the edge is a single straight
line: the table gives a strength with no path, and a strength has no shape.

For subject 36 that is 99 parcels in the tumor and 46 tied to them, over 259
connections and 101 spokes, laid out in a fifth of a second. Only connections
at or above 15% of the strongest are drawn, or the chart is a solid mat.

The modality strip sits in the viewer's bottom border, away from the other
controls on the right. Press `1` to `4` or click it. The number of the one in
use is bright, and anything the current space does not carry is grayed.

The cursor panel gives every sequence's intensity at that voxel, not just the
one on screen, and in MNI it names the parcel under the crosshair.

MNI also draws the involved parcels onto the slices, in the same colors the
3D uses, under the tumor rather than over it. The selected one is brighter.
That is why `p` works there and not in native space: native has no parcels to
color, so the title says which palette is on only where it means something.

```
v  3d / slices          1-4  modality           t  focus the prompt
s  native / mni         a b c  breakdown view    x  cancel the run
p  parcel coloring                              q  quit
```

Half of those only make sense in one of the two viewer modes. The ones that do
not apply go gray in the footer rather than doing nothing silently.

Each panel carries a strip along its bottom left of the things it can show —
the four modalities in the viewer, the three breakdown views — with the key for
each one and the current one bright. The rest of the controls sit bottom right,
labeled with what they give you rather than with what you are looking at, so
in native space `s` reads `s mni`.

Each parcel row carries a chip in that parcel's own color, so a shape you can
see in the 3D can be found in the list without clicking through it. The chips
follow the palette toggle.

The layer checkboxes drive both views: unticking `SNFH edema` takes it out of
the 3D and out of the slice overlay, so there is no separate overlay key.

`tumor` above them is the three classes contoured as one closed surface rather
than three that meet along internal faces, with its parts indented under it. It
is a real parent: unticking it takes the parts with it, and it draws its own
combined shape only when none of the parts is on, so the two can never wash
each other out.

A rule separates the tumor and its margin from the tracts and the brain
surface, which are context rather than parts of it.

For a patient with two timepoints, a second rule divides off three more: what both scans have in cream, what is new in the later one in
red, and what has gone since the first in green. They draw the same tissue the
tumor layers do, so the two groups are exclusive: ticking one of these unticks
the tumor, and ticking the tumor unticks these. They are computed in MNI, the only frame the two share, and they are the shapes
behind the Dice in `changes`. On the slices they draw in either space: in
native they come back through this subject's own inverse warp, so they sit on
the scan rather than on the atlas. Round tripping a segmentation through that
inverse returns it at Dice 0.996.

The volumes differ between the two, 61 mL of shared tumor in MNI against 54 in
native, because the warp stretches the brain. The native figure is the
patient's.

Every layer draws in both views. The margin is an outline on the slices and the
rest are fills, so anything you tick shows up whichever viewer you are in.

`margin 2 cm` above that is the whole tumor grown by 20 mm, the usual CTV
expansion, **clipped to the brain**: a fifth to a quarter of a raw expansion
lands in skull and air, which would put subject 36 at 751 mL instead of 549.
It is an outline on the slices rather than a fill, since 2 cm of fill hides the
thing you grew it from. Both the surface and the slice mask are built the first
time you tick it, because the distance transform is not free.

The volume is measured in native space, where the millimeters are the
patient's, not in the warped MNI copy. Across these six subjects a 2 cm margin
is 3.3 to 7.5 times the tumor it came from, the multiplier being largest for
the smallest tumors.

The breakdown lists the layers with their volumes, clickable to show and hide,
then one of three views: the involved parcels, the functions at risk, or the
connectivity at risk. Whatever is selected
lights up in the 3D and every other parcel drops back, so a row and a shape in
the viewer are the same thing. One row means one parcel: picking it draws that
parcel and nothing else, in the 3D and on the slices both. There is no way to
show them all at once, because a hundred overlapping ribbons is not a picture
of anything.

The other two views select more than one, because their rows are about more
than one: a function row draws the parcels behind its score, and a connectivity
row draws the untouched parcel together with the involved ones it is losing.

The parcel list follows the layers: it shows the parcels the visible layers are
actually drawing. Untick `ET enhancing` and the parcels that are mostly
enhancing leave the list; tick `new in the later` and it becomes the parcels
that piece of the recurrence touches.

In the slice view, picking a parcel puts the crosshair inside it instead. Only
MNI knows where the parcels are, so it switches space if it has to. It works both ways: click a parcel in the 3D and
its row is selected, drag and it rotates instead.

WT, TC and ET are unions of the tissue classes, not classes themselves, which
is why none of their volumes matches a single layer. The inputs panel spells
out what each one contains and leaves them uncolored, since the three colors
belong to the classes in the breakdown.

Under them the panel carries `where` the tumor sits, by hemisphere and lobe,
which is the same question the 119 parcel rows answer one at a time, and `run`,
which segmenter produced this and on what, how long it took, its Dice where
there are reference labels, and `aligned on`: the share of the brain the
registration had left to work with once the lesion was masked out of the metric.
That last one is about a single registration, not about change over time.

Click a subject or focus the panel and arrow through them; it scrolls to keep
the chosen one in view, and steps over any whose outputs have been deleted.

The panel is in two parts. `stats` holds the tumor, where it sits and how it
was made; the lobe volumes are indented under `in parcels` because that is the
volume they add up to, and they sit under a hemisphere only when the tumor is
in both, since otherwise the hemisphere is the whole of it.

There are four tissue classes but only three of them usually appear. `RC`, the
resection cavity, is a post-treatment class, so its region row and its layer
show up only on a subject that has one.

Between them, `changes` appears when a patient has more than one timepoint: the
change in WT, how many parcels the tumor newly reaches or has left, the Dice
between the two tumors, and how much of the later one sits inside the earlier
one's footprint. Subject 36 comes back at Dice 0.56 with 91% of the recurrence
inside where the original was.

Those two are not the same question. Dice is low because the tumor shrank, not
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
of it the tumor occupies and what that contributed; those shares add up to the
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
function terms. For a tumor it sums those profiles, weighting each involved
parcel by how much of it the tumor occupies.

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
tumor has **not** touched, sums the connections it has into parcels the tumor
**has**, weighted by how much of each of those is gone.

That answers a question the overlap cannot: which intact regions are losing
their input. Selecting one lights it in the 3D and dims everything except the
involved parcels it connects to, and the panel underneath lists those with how
much of the loss each accounts for.

```
L_SM2_B   Left supramarginal gyrus 2 B, parietal, not in the tumor
          25% of its connectivity at risk
          L_SM1_D  52% involved, 18.8% of the loss
          L_SM1_E  71% involved, 18.7% of the loss
```

This is a **healthy population mean**, the same caveat as Parcelsynth. A tumor
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
tracts.py      tract centerlines
view.py        3D view, also runnable on its own
tui.py         terminal UI
render.py      offscreen 3D and slice images for the TUI
function.py    Parcelsynth terms
paths.py       shared paths
```

## Worth knowing

The faint gray brain is one closed surface contoured out of MNI152, not the 696
parcel meshes. Those are 754 separate closed bodies, so stacking them as a
translucent shell piles up alpha along every view ray until the involved
parcels inside end up darker than the context around them.

If you have an NVIDIA GPU it uses BraTS25_1, which won the 2025 challenge. On
CPU it falls back to BraTS25_2 and says so. Nothing to configure either way.

main.py has a shebang pointing at .venv, so ./main.py works without activating
anything. Fix that line if you move the project.
