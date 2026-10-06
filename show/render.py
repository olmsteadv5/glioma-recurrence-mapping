# Offscreen rendering for the terminal UI: 3D scenes and slice planes, PIL images out.

import functools
import os
import subprocess

import nibabel as nib
import numpy as np
import pyvista as pv
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkWindowToImageFilter
from PIL import Image, ImageDraw

from show import view
from show.compare import _inverse
from yale import atlas
from hcp import streamlines
from geometry import outline
from paths import (LABELS, MNI, MODALITIES, THREADS, cache_dir, modality, out_dir, pair_cache,
                   pair_dir, raw_dir, stripped_dir, walk_dir)

pv.OFF_SCREEN = True



def _panel(base=(40, 40, 40)):
    # kitty draws cell backgrounds at background_opacity but graphics fully
    # opaque, so an image painted #282828 comes out lighter than the panels
    # around it. paint it the color those panels actually composite to.
    where = os.environ.get("KITTY_CONFIG_DIRECTORY", os.path.expanduser("~/.config/kitty"))
    opacity = 1.0
    try:
        for line in open(os.path.join(where, "kitty.conf")):
            if line.split()[:1] == ["background_opacity"]:
                opacity = min(max(float(line.split()[1]), 0.0), 1.0)
    except (OSError, ValueError, IndexError):
        pass
    # the compositor behind the window is black, so the blend is a plain scale
    return tuple(round(c * opacity) for c in base)


PANEL = _panel()
BG = "#%02x%02x%02x" % PANEL
SHELL = (168, 153, 132)
# tracts through the pre-op tumor, the follow-up's, and both; both is the cream
# the compare layers use for tissue in both scans
TRACTS = {"1": (198, 124, 72), "2": (211, 134, 155), "both": (235, 219, 178)}
TISSUE = {1: (242, 193, 78), 2: (63, 167, 214), 3: (228, 87, 46), 4: (142, 125, 190),
          0: (214, 200, 175)}  # 0 is the three of them as one shape
CROSS = (250, 189, 47)
LIT = (251, 241, 199)
WHOLE = "WT"
COMPARE = {"both": (235, 219, 178), "new": (251, 73, 52), "gone": (142, 192, 124)}
# the targets from analysis/walkdown.py, one per NRG phase: the walk-down, and the
# NRG 2 cm volume it is scored against, exactly as the tables score it
WALK = {"46": (131, 165, 152), "60": (215, 153, 33)}
NRG = {"46": (177, 98, 134), "60": (126, 108, 204)}
WALK_PHASES = tuple(WALK)
TARGETS = {"walk": WALK, "nrg": NRG}
# the layer key each kind of target goes by
TARGET_KEY = {"walk": "W:", "nrg": "T:"}


def target_file(subject, kind, phase, space):
    # where a target sits for this scan in this space. Both scans of a pair show
    # the one target, which is built from the pre-op tumor. None if
    # analysis/walkdown.py has not been run for this patient
    mni = walk_dir(subject) / f"{kind}_{phase}_mni.nii.gz"
    if not mni.exists():
        return None
    if space == "mni":
        return mni
    if subject.endswith("_21"):
        here = walk_dir(subject) / f"{kind}_{phase}.nii.gz"
        # walkdown.py writes the follow-up's copy itself
        return here if here.exists() else None
    here = cache_dir(subject) / f"{kind}_{phase}.nii.gz"
    if not here.exists() or here.stat().st_mtime < mni.stat().st_mtime:
        # the pre-op scan's own grid, through its own way back from MNI
        subprocess.run(["greedy", "-d", "3", "-rf", str(modality(stripped_dir(subject), subject, "T1")),
                        "-ri", "LABEL", "0.2vox", "-rm", str(mni), str(here),
                        "-r", str(out_dir(subject) / "affine.mat") + ",-1", str(_inverse(subject)),
                        "-threads", THREADS], capture_output=True)
    return here if here.exists() else None


def walk_ends(subject):
    # where each walk stopped, both phases, in MNI mm
    out = {}
    for phase in WALK_PHASES:
        path = walk_dir(subject) / f"walk_{phase}_ends.npy"
        if path.exists():
            out[phase] = np.load(path)
    return out
_hex = lambda t: "#%02x%02x%02x" % tuple(t)


def fit(image, size, background=PANEL):
    # Always hand back exactly the requested box, so the widget never shows dead space.
    w, h = max(size[0], 8), max(size[1], 8)
    scale = min(w / image.width, h / image.height)
    scaled = image.resize((max(int(image.width * scale), 1), max(int(image.height * scale), 1)),
                          Image.NEAREST)
    canvas = Image.new("RGB", (w, h), background)
    canvas.paste(scaled, ((w - scaled.width) // 2, (h - scaled.height) // 2))
    return canvas


class Scene:
    # One plotter per subject; between frames only the camera moves.

    def __init__(self, subject, overlap, palette="contrast", size=(700, 540), later=None):
        # subject is the scan plans are made from; later, when given, is the same
        # patient's follow-up, drawn alongside it
        self.subject, self.later, self.overlap, self.palette = subject, later, overlap, palette
        self.plotter = pv.Plotter(off_screen=True, window_size=list(size))
        self.plotter.set_background(BG)
        self.actors, self.parcel_actors = {}, {}

        info, surf = atlas.parcels(), atlas.meshes()
        index = {p["name"]: i for i, p in enumerate(info)}
        colors = atlas.colors(palette)
        involved = [r["parcel"] for r in overlap["parcels"]]

        self.actors["shell"] = self._mesh(*atlas.brain(), _hex(SHELL), 0.15)

        for name in involved:
            i = index[name]
            actor = self._mesh(surf[i].vertices, surf[i].faces, _hex(colors[i]), 1.0)
            self.parcel_actors[name] = actor

        # the follow-up sees through, so the pre-op tumor shows under it
        self.segs = {}
        for scan, who in (("1", subject), ("2", later)):
            if who is None:
                continue
            image = nib.load(out_dir(who) / "seg_in_mni.nii.gz")
            self.segs[scan] = image
            for value, (name, points, faces) in self._tumor(who, image).items():
                solid = 0.5 if value == 0 else 1.0
                self.actors[f"{scan}:{name}"] = self._mesh(points, faces, _hex(TISSUE[value]),
                                                           solid if scan == "1" else solid * 0.4)

        self._crossing = self._through()
        self.tract_count, self.tract_keys = {}, None

        self._picker, self.lit, self.shown = None, None, None
        self.plotter.show(auto_close=False)

        # nothing is watching this window, so never wait on its vertical blank.
        # pyvista's own screenshot reads the front buffer, which only updates on
        # the swap, so read the back buffer instead: 17 ms a frame down to 3.
        window = self.plotter.ren_win
        window.SetSwapBuffers(0)
        # multisampling off: VTK only resolves the back buffer to match the
        # front one when it is off, and the edges look the same at this size
        window.SetMultiSamples(0)
        self.grab = vtkWindowToImageFilter()
        self.grab.SetInput(window)
        self.grab.FixBoundaryOn()
        self.grab.ShouldRerenderOff()
        self.grab.SetInputBufferTypeToRGB()
        self.grab.ReadFrontBufferOff()

    @staticmethod
    def _tumor(subject, image):
        # contouring the four labels is half a second, and it only changes
        # when the registration does
        cache = cache_dir(subject) / "tumor_surfaces.npz"
        if cache.exists() and cache.stat().st_mtime >= os.path.getmtime(image.get_filename()):
            held = np.load(cache)
            return {int(v): (WHOLE if int(v) == 0 else LABELS[int(v)],
                             held[f"v{v}"], held[f"f{v}"]) for v in held["values"]}
        data = np.asarray(image.dataobj).astype(np.int16)
        out, saved = {}, {"values": []}
        # value 0 is all three classes contoured as one closed surface
        for value, name in list(LABELS.items()) + [(0, WHOLE)]:
            source = (data > 0).astype(np.int16) if value == 0 else data
            surface = view._tumor_surface(source, image.affine, 1 if value == 0 else value)
            if surface:
                out[value] = (name, surface[0], surface[1])
                saved["values"].append(value)
                saved[f"v{value}"], saved[f"f{value}"] = surface
        saved["values"] = np.array(saved["values"])
        np.savez_compressed(cache, **saved)
        return out

    def _mesh(self, v, f, color, opacity):
        f = np.asarray(f)
        poly = pv.PolyData(np.asarray(v), np.hstack([np.full((len(f), 1), 3), f]).ravel())
        return self.plotter.add_mesh(poly, color=color, opacity=opacity, smooth_shading=True)

    def _through(self):
        # for each scan and tissue class, the tracts with a point in it, so the
        # tracts drawn can follow whatever tumor is on without a fresh search
        lines = streamlines.streamlines()
        stacked = np.vstack(lines)
        owner = np.repeat(np.arange(len(lines)), [len(l) for l in lines])
        out = {}
        for scan, image in self.segs.items():
            data = np.asarray(image.dataobj).astype(np.int16)
            ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine), stacked)).astype(int)
            ok = np.all((ijk >= 0) & (ijk < data.shape), axis=1)
            value = np.zeros(len(stacked), np.int16)
            value[ok] = data[tuple(ijk[ok].T)]
            for label in LABELS:
                out[(scan, label)] = set(np.unique(owner[value == label]).tolist())
        return out

    def show_crossing(self, scans):
        # the streamlines through each chosen scan's whole tumor, whatever tissue
        # is on screen. With both scans chosen, a streamline through both is
        # drawn once, in its own color, so the two never fight over one place
        scans = frozenset(scans)
        if scans == self.tract_keys:
            return
        self.tract_keys = scans
        through = {scan: set().union(*[self._crossing.get((scan, value), set()) for value in LABELS])
                   if scan in scans else set() for scan in ("1", "2")}
        shared = through["1"] & through["2"]
        groups = {"1": through["1"] - shared, "2": through["2"] - shared, "both": shared}
        self.tract_count = {"1": len(through["1"]), "2": len(through["2"]), "both": len(shared)}
        lines = streamlines.streamlines()
        for group, ids in groups.items():
            key = f"tracts:{group}"
            if key in self.actors:
                self.plotter.remove_actor(self.actors.pop(key))
            mesh = self._lines([lines[i] for i in sorted(ids)])
            if mesh is not None:
                self.actors[key] = self.plotter.add_mesh(
                    mesh, color=_hex(TRACTS[group]), line_width=1,
                    opacity=self.TRACT_DIM if self.shown else 0.55)

    @staticmethod
    def _lines(lines):
        if not lines:
            return None
        points, cells, offset = [], [], 0
        for line in lines:
            points.append(line)
            cells.append(np.hstack([[len(line)], np.arange(offset, offset + len(line))]))
            offset += len(line)
        return pv.PolyData(np.vstack(points), lines=np.hstack(cells))

    def resize(self, width, height):
        size = [max(width, 64), max(height, 64)]
        if size != list(self.plotter.window_size):
            self.plotter.window_size = size

    def rotate(self, azimuth, elevation):
        # the camera.azimuth property does not rotate; these VTK methods do
        self.plotter.camera.Azimuth(azimuth)
        self.plotter.camera.Elevation(elevation)
        self.plotter.camera.OrthogonalizeViewUp()

    def zoom(self, factor):
        self.plotter.camera.Zoom(factor)

    def set_visible(self, key, on):
        actor = self.actors.get(key)
        if actor is not None:
            actor.visibility = bool(on)

    def set_opacity(self, key, value):
        actor = self.actors.get(key)
        if actor is not None:
            actor.prop.opacity = float(value)

    def only(self, names):
        # one parcel at a time, or the handful a row is actually about
        keep = {names} if isinstance(names, str) else set(names)
        for other, actor in self.parcel_actors.items():
            actor.visibility = other in keep
            actor.prop.opacity = 1.0

    def ensure_target(self, kind, phase):
        # built the first time it is asked for, and cached on the mask's own date
        # so a rebuilt target never shows the old shape
        key = TARGET_KEY[kind] + phase
        if key in self.actors:
            return
        source = target_file(self.subject, kind, phase, "mni")
        if source is None:
            return
        cache = pair_cache(self.subject) / f"{kind}_{phase}_surface.npz"
        if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
            held = np.load(cache)
            points, faces = held["vertices"], held["faces"]
        else:
            image = nib.load(source)
            surface = view._tumor_surface(np.asarray(image.dataobj).astype(np.int16), image.affine, 1)
            if surface is None:
                return
            points, faces = surface
            np.savez_compressed(cache, vertices=points, faces=faces)
        self.actors[key] = self._mesh(points, faces, _hex(TARGETS[kind][phase]), 0.22)

    def ensure_ends(self, phase):
        # one dot where each walk of this phase stopped, in its walk-down's color
        key = "W:ends" + phase
        if key in self.actors:
            return
        ends = walk_ends(self.subject).get(phase)
        if ends is None or not len(ends):
            return
        self.actors[key] = self.plotter.add_mesh(pv.PolyData(ends), color=_hex(WALK[phase]),
                                                 point_size=4, render_points_as_spheres=True)

    def ensure_compare(self, first, second):
        # the two timepoints against each other, built the first time it is asked for
        if "C:both" in self.actors:
            return
        from show import compare
        for name, (points, faces) in compare.surfaces(first, second).items():
            self.actors["C:" + name] = self._mesh(points, faces, _hex(COMPARE[name]), 0.6)

    TRACT_DIM = 0.12

    def show_tracts(self, ids):
        # whatever is picked in the chart, picked out of the bundle here: one
        # tract from a row of its panel, or every tract of a parcel
        ids = frozenset(ids) if ids else None
        if ids == self.shown:
            return
        if self.shown is not None:
            self.plotter.remove_actor(self.actors.pop("tract"))
        self.shown = ids
        # the rest of the bundle falls back so the picked ones are readable
        for group in TRACTS:
            self.set_opacity(f"tracts:{group}", self.TRACT_DIM if ids else 0.55)
        if ids is None:
            return
        lines = streamlines.streamlines()
        mesh = self._lines([lines[n] for n in sorted(ids) if n < len(lines)])
        if mesh is None:
            return
        self.actors["tract"] = self.plotter.add_mesh(mesh, color=_hex(LIT),
                                                     line_width=3)

    def spotlight(self, name):
        # an untouched parcel is not in the scene, so add it only when asked for
        if name == self.lit:
            return
        if self.lit is not None:
            self.plotter.remove_actor(self.actors.pop("lit"))
            self.lit = None
        index = {p["name"]: i for i, p in enumerate(atlas.parcels())}
        if name not in index:
            return
        mesh = atlas.meshes()[index[name]]
        self.actors["lit"] = self._mesh(mesh.vertices, mesh.faces, _hex(LIT), 1.0)
        self.lit = name

    def pick(self, x, y):
        # ray cast against the parcels only, so the shell cannot intercept
        if self._picker is None:
            self._picker = vtkCellPicker()
            self._picker.SetTolerance(0.001)
            for actor in self.parcel_actors.values():
                self._picker.AddPickList(actor)
            self._picker.PickFromListOn()
        height = self.plotter.window_size[1]
        if not self._picker.Pick(x, height - 1 - y, 0, self.plotter.renderer):
            return None
        hit = id(self._picker.GetActor())
        return next((n for n, a in self.parcel_actors.items() if id(a) == hit), None)

    def recolor(self, palette):
        self.palette = palette
        colors = atlas.colors(palette)
        index = {p["name"]: i for i, p in enumerate(atlas.parcels())}
        for name, actor in self.parcel_actors.items():
            actor.prop.color = _hex(colors[index[name]])

    def snapshot(self, width=1600, height=1200):
        # bigger than the pane, for something you are going to send to someone
        was = list(self.plotter.window_size)
        self.resize(width, height)
        out = self.image()
        self.resize(*was)
        return out

    def image(self):
        self.plotter.render()
        self.grab.Modified()
        self.grab.Update()
        data = self.grab.GetOutput()
        width, height, _ = data.GetDimensions()
        pixels = vtk_to_numpy(data.GetPointData().GetScalars())
        return Image.fromarray(pixels.reshape(height, width, 3)[::-1])


@functools.lru_cache(maxsize=4)
def _font(size):
    # the terminal's own face, so drawn text reads as part of the interface
    from PIL import ImageFont
    for where in ("/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
                  "/usr/share/fonts/TTF/NotoSansMono-Regular.ttf",
                  "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(where, size)
        except OSError:
            continue
    return ImageFont.load_default()


@functools.lru_cache(maxsize=8192)
def _stamp(text, size, fill, anchor):
    # PIL rasterises a string from scratch every time it is drawn, and the
    # tract panel writes a hundred short ones a frame. Keep each as a tile.
    font = _font(size)
    box = font.getbbox(text, anchor=anchor)
    tile = Image.new("RGBA", (max(box[2] - box[0], 1), max(box[3] - box[1], 1)),
                     (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((-box[0], -box[1]), text, font=font,
                              fill=tuple(fill) + (255,), anchor=anchor)
    return tile, box[0], box[1]


class Ink:
    # an ImageDraw whose text() pastes a cached tile instead of rasterising

    def __init__(self, canvas):
        self.canvas, self.pen = canvas, ImageDraw.Draw(canvas)

    def __getattr__(self, name):
        return getattr(self.pen, name)

    def text(self, xy, text, font=None, fill=None, anchor="la"):
        if not text or fill is None or font is None:
            return self.pen.text(xy, text, font=font, fill=fill, anchor=anchor)
        tile, dx, dy = _stamp(str(text), font.size, tuple(fill), anchor)
        self.canvas.paste(tile, (int(xy[0]) + dx, int(xy[1]) + dy), tile)


NODE_IN = (228, 87, 46)
NODE_OUT = (132, 186, 108)
NODE_TUMOR = (250, 189, 47)
# one hue per kind of link: cool where the mass is not in the path, warm where
# it is. strength brightens a line without draining its hue
SPOKE = (152, 100, 108)
THROUGH = (154, 112, 72)
STRAY = (108, 170, 232)
CLIP = (96, 96, 208)
# the interface palette, so the key is the same grays as every other panel
INK = (168, 153, 132)
FAINT_INK = (124, 111, 100)
BRIGHT_INK = (235, 219, 178)
KEY_EDGE = (80, 73, 69)
# one ground per hemisphere and lobe. the lobe picks the hue and the hemisphere
# how much of it there is. the lightness is held equal in CIELAB, not in rgb,
# because the same rgb value is a much lighter yellow than it is a blue, which
# is what made the right hand regions read as a fainter version of the left
LOBE = {"Frontal": 268, "Parietal": 150, "Temporal": 40, "Occipital": 310,
        "Insula": 80, "Corpus callosum": None}


@functools.lru_cache(maxsize=64)
def _ground(side, lobe, lift=0.62):
    import math
    light = 62.0 if lift > 1.0 else 26.0
    turn = LOBE.get(lobe)
    chroma = 0.0 if turn is None else (34.0 if side == "Left" else 15.0)
    a = chroma * math.cos(math.radians(turn or 0))
    b = chroma * math.sin(math.radians(turn or 0))

    y = (light + 16) / 116
    x, z = y + a / 500, y - b / 200
    undo = lambda t: t ** 3 if t ** 3 > 0.008856 else (t - 16 / 116) / 7.787
    x, y, z = undo(x) * 0.95047, undo(y), undo(z) * 1.08883
    out = (x * 3.2406 - y * 1.5372 - z * 0.4986,
           -x * 0.9689 + y * 1.8758 + z * 0.0415,
           x * 0.0557 - y * 0.2040 + z * 1.0570)
    gamma = lambda u: 12.92 * u if u <= 0.0031308 else 1.055 * u ** (1 / 2.4) - 0.055
    return tuple(int(round(255 * min(max(gamma(max(v, 0.0)), 0.0), 1.0))) for v in out)


class Chart:
    # The tumor's neighborhood as a node chart, laid out so the gaps between
    # nodes are the real distances between the parcels.

    def __init__(self, subject, overlap):
        from yale import graph
        self.subject = subject
        (self.nodes, self.link, self.far, self.inside, self.spokes,
         self.void, self.crossed) = graph.build(subject, overlap)
        info = atlas.parcels()
        self.names = [p["name"] for p in info]
        self.lobes = [(info[n]["hemisphere"], info[n]["lobe"]) for n in self.nodes]

        self.about = {r["parcel"]: r for r in overlap["parcels"]}
        self.detail = {p["name"]: p for p in info}
        lost = {r["parcel"]: r["in_tumor_ml"] for r in overlap["parcels"]}
        self.lost = [lost.get(self.names[n], 0.0) for n in self.nodes]
        self.most = max(self.lost, default=1.0) or 1.0
        # no edge and no spoke reaches these, so they are drawn hollow rather
        # than left looking like the layout dropped them
        self.loose = (set(range(len(self.nodes)))
                      - {e[0] for e in self.link} - {e[1] for e in self.link}
                      - {s[0] for s in self.spokes})
        self.tract = None                     # the one tract picked, if any

        # one line per connection, never one per tract: the tract count is the
        # line's weight, so the picture stays a connectome and not a hairball
        # every mark the key lists, so its rows sum to its own total
        self.strands = len(self.link) + len(self.spokes) + len(self.void)
        self.tally = {
            "in": len(self.inside), "out": len(self.nodes) - len(self.inside),
            "mass": 1, "loose": len(self.loose),
            "pair": (len(self.link), sum(e[2] for e in self.link)),
            "spoke": (len(self.spokes), sum(e[1] for e in self.spokes)),
            "gone_mass": (sum(1 for at, *_ in self.void if at is None),
                          sum(many for at, many, *_ in self.void if at is None)),
            "gone_parcel": (sum(1 for at, *_ in self.void if at is not None),
                            sum(many for at, many, *_ in self.void if at is not None))}

        # one wedge per region, so the ring order is anatomy and not the seed
        self.around = sorted(range(len(self.nodes)),
                             key=lambda n: (self.lobes[n], self.names[self.nodes[n]]))
        self.slot = {n: k for k, n in enumerate(self.around)}

        self._runs_was, self._runs, self._rows_at = object(), [], None
        self.at, self.fits = 0, 1
        self._body_was, self._body = None, None

        self.lit, self.guest, self.zoom, self.pan = None, None, 1.0, np.zeros(2)
        self.crowd, self.label = set(), None
        self._key_was, self._key_art = None, None
        self.where = {}
        self.screen = np.zeros((0, 2))
        self.middle, self.seat = np.zeros(2), {}
        self.wide, self.tall = 1.0, 1.0
        # the tumor, then the hub for every far end the atlas cannot name
        self.hub, self.gone = len(self.nodes), len(self.nodes) + 1

    def name(self, at):
        if at is None or at >= len(self.nodes):
            return None
        return self.names[self.nodes[at]]

    def picked(self):
        # what the pick covers: one parcel, or the several a function is made of
        if self.crowd:
            return set(self.crowd)
        return {self.lit} if self.lit is not None else set()

    def gather(self, parcels, label):
        # a function is a weighted sum over parcels, so it lights all of them
        self.lit = self.guest = None
        seat = {self.names[n]: k for k, n in enumerate(self.nodes)}
        self.crowd = {seat[p] for p in parcels if p in seat}
        self.label = label if self.crowd else None

    def focus(self, parcel):
        # the node for the picked row: the one already drawn, or a guest placed
        # by its real distances, for a row naming a parcel too lightly touched
        # to have earned a node of its own
        from yale import graph
        self.lit = self.guest = self.label = None
        self.crowd = set()
        if parcel is None:
            return
        self.lit = next((n for n in range(len(self.nodes))
                         if self.names[self.nodes[n]] == parcel), None)
        if self.lit is not None:
            return
        made = graph.attach(self.subject, self.nodes, parcel)
        if made is None:
            return
        n, link, far, lengths, ids = made
        info = atlas.parcels()[n]
        self.guest = {"name": parcel, "link": link, "far": far,
                      "lengths": lengths, "ids": ids,
                      "lobe": (info["hemisphere"], info["lobe"]),
                      "screen": np.zeros(2)}

    def move(self, zoom=1.0, pan=(0.0, 0.0)):
        self.zoom = float(np.clip(self.zoom * zoom, 0.4, 12.0))
        self.pan = self.pan + np.array(pan, dtype=float)

    def home(self):
        self.zoom, self.pan = 1.0, np.zeros(2)

    def _place(self, wide, high):
        # parcels on a rim grouped by region, the two hubs abreast in the middle.
        # an ellipse because the pane is wider than it is tall and two hubs want
        # the width. nothing here is a distance, so nothing pretends to be one
        self.middle = np.array([wide, high]) / 2 + self.pan
        # the pane can be as wide as the monitor is, but the ellipse only wants
        # enough width for two hubs abreast. past that it stops stretching and
        # the extra goes to the margins instead of to the shape
        self.tall = max(high / 2 - 74, 40) * self.zoom
        self.wide = min(max(wide / 2 - 104, 40) * self.zoom, self.tall * self.STRETCH)
        step = 2 * np.pi / max(len(self.nodes), 1)

        turn = np.array([self.slot[n] * step - np.pi / 2 for n in range(len(self.nodes))])
        self.screen = np.zeros((len(self.nodes) + 2, 2))
        self.screen[:len(self.nodes)] = self.middle + np.stack(
            [self.wide * np.cos(turn), self.tall * np.sin(turn)], axis=1)
        apart = self.wide * 0.3
        self.screen[self.hub] = self.middle - [apart, 0]
        self.screen[self.gone] = self.middle + [apart, 0]

        # one seat per region, half way in, so a thread joins two regions
        # before it joins two parcels and parallel tracts travel together
        self.seat = {}
        for lobe in sorted(set(self.lobes)):
            slots = [self.slot[n] for n in range(len(self.nodes)) if self.lobes[n] == lobe]
            mean = (min(slots) + max(slots)) / 2 * step - np.pi / 2
            self.seat[lobe] = self.middle + 0.52 * np.array(
                [self.wide * np.cos(mean), self.tall * np.sin(mean)])
        if self.guest is not None:
            near = [turn[k] for k, *_ in self.guest["link"]] or [0.0]
            mean = np.arctan2(np.mean(np.sin(near)), np.mean(np.cos(near)))
            self.guest["screen"] = self.middle + np.array(
                [(self.wide + 26) * np.cos(mean), (self.tall + 26) * np.sin(mean)])

    def image(self, size):
        wide, high = max(size[0], 64), max(size[1], 64)
        if not len(self.nodes):
            return Image.new("RGB", (wide, high), PANEL)
        # scrolling the panel changes nothing about the chart under it, so the
        # chart is kept and only the panel is drawn again
        seen = (wide, high, self.zoom, tuple(self.pan), self.lit, self.tract,
                tuple(sorted(self.crowd)),
                self.guest["name"] if self.guest else None)
        if seen == self._body_was:
            canvas = self._body.copy()
            self._place(wide, high)
            self._shelf(Ink(canvas), wide, high)
            return canvas
        canvas = Image.new("RGB", (wide, high), PANEL)
        pen = Ink(canvas)
        self._place(wide, high)
        near = self._neighbors()
        self._bands(pen)
        # a line stays lit only if the picked node is one of its two ends. the
        # neighbors themselves stay lit, but the lines between them do not,
        # which used to read as the picked parcel reaching further than it does
        free = near is None
        picked = self.picked()
        mine = lambda ids: self.tract is not None and self.tract in ids
        # the dim ones first and the picked ones over them, so a pick is never
        # buried under the lines it was picked out of
        for lit in (False, True):
            for at, many, lengths, ids in self.spokes:
                own = at in picked or mine(ids)
                if own == lit:
                    self._join(pen, at, self.hub, many, lengths, self.far[at],
                               self._shade(SPOKE, free or own), own)
            for at, many, lengths, ids in self.void:
                own = at in picked or mine(ids)
                if own == lit:
                    tone = STRAY if at is None else CLIP
                    self._join(pen, self.hub if at is None else at, self.gone,
                               many, lengths, 60.0, self._shade(tone, free or own), own)
            for a, b, many, gap, lengths, ids in self.link:
                own = a in picked or b in picked or mine(ids)
                if own == lit:
                    self._join(pen, a, b, many, lengths, gap,
                               self._shade(THROUGH, free or own), own)
        if self.guest is not None:
            for k, many, gap, lengths, ids in self.guest["link"]:
                self._join(pen, k, "guest", many, lengths, gap, THROUGH, True)
        hot = self._hot()
        self._nodes(pen, near, hot, picked)
        if self.guest is not None:
            self._guest(pen)
        self._hubs(pen, near, hot)
        self._labels(pen, near)
        self._key(canvas, pen, wide, high)
        self._body_was, self._body = seen, canvas.copy()
        self._shelf(pen, wide, high)
        return canvas

    def _guest(self, pen):
        # nothing of it is gone, so it draws at the smallest size the key shows
        point, wide = self.guest["screen"], 3.0
        pen.ellipse([point[0] - wide, point[1] - wide, point[0] + wide, point[1] + wide],
                    fill=NODE_OUT, outline=LIT, width=3)
        pen.text((point[0], point[1] - 14), self.guest["name"], fill=LIT,
                 font=_font(11), anchor="ms")

    def _bands(self, pen):
        # a region is a run of the rim, so it is contiguous by construction
        step = 360.0 / max(len(self.nodes), 1)
        box = [self.middle[0] - self.wide - 24, self.middle[1] - self.tall - 24,
               self.middle[0] + self.wide + 24, self.middle[1] + self.tall + 24]
        self.where = {}
        for lobe in sorted(set(self.lobes)):
            slots = [self.slot[n] for n in range(len(self.nodes)) if self.lobes[n] == lobe]
            pen.arc(box, min(slots) * step - 90 - step * 0.4,
                    max(slots) * step - 90 + step * 0.4,
                    fill=_ground(*lobe, lift=1.5), width=13)
            self.where[lobe] = None

    THREAD = np.linspace(0, 1, 26)[:, None, None]

    def _ribbon(self, pen, a, seat_a, b, seat_b, many, color, width=1):
        # one thread per tract, drawn through both regions on its way across.
        # they are spread a little so a bundle reads as a ribbon, and its
        # thickness is how many tracts there are, not how long any of them is
        across = b - a
        span = float(np.linalg.norm(across))
        side = (np.array([-across[1], across[0]]) / span if span > 1
                else np.array([0.0, 0.0]))
        middle = self.middle * 0.6 + (seat_a + seat_b) * 0.2
        for n in range(many):
            off = side * (n - (many - 1) / 2) * 3.0
            knots = np.array([a, seat_a + off, middle + off, seat_b + off, b])
            q = np.repeat(knots[None], len(self.THREAD), 0)
            for _ in range(4):
                q = q[:, :-1] * (1 - self.THREAD) + q[:, 1:] * self.THREAD
            pen.line([tuple(v) for v in q[:, 0]], fill=color, width=width)

    # how far the ellipse may run wider than it is tall, however wide the pane
    STRETCH = 1.85
    SHELF = 392
    GUEST = object()
    STEPS_MM = (10, 20, 25, 50, 100, 200, 500)

    def _dot(self, n):
        # the size and color that node has on the chart
        if n is None:
            return 7.0, NODE_TUMOR, PANEL
        if n is self.GUEST:
            return 3.0, NODE_OUT, LIT
        if n >= len(self.nodes):
            # a hub, which is not a parcel and has no volume to size it by
            return (7.0, NODE_TUMOR, PANEL) if n == self.hub else (6.0, STRAY, PANEL)
        size = 3.0 + 7.0 * (self.lost[n] / self.most) ** 0.5
        fill = NODE_IN if self.nodes[n] in self.inside else NODE_OUT
        return (size, PANEL, fill) if n in self.loose else (size, fill, PANEL)

    def runs(self):
        # every tract of whatever is picked, longest first, each carrying the
        # id of the streamline itself so the 3d can be asked to show that one
        key = (self.lit, tuple(sorted(self.crowd)),
               self.guest["name"] if self.guest else None)
        if key == self._runs_was:
            return self._runs
        out = []
        if self.guest is not None:
            for k, many, gap, lengths, ids in self.guest["link"]:
                out += [(mm, THROUGH, self.GUEST, k, t) for mm, t in zip(lengths, ids)]
            out += [(mm, SPOKE, self.GUEST, None, t)
                    for mm, t in zip(self.guest["lengths"], self.guest["ids"])]
        else:
            picked = self.picked()
            for a, b, many, gap, lengths, ids in self.link:
                if picked and not (a in picked or b in picked):
                    continue
                near, far = (b, a) if b in picked and a not in picked else (a, b)
                out += [(mm, THROUGH, near, far, t) for mm, t in zip(lengths, ids)]
            for at, many, lengths, ids in self.spokes:
                if picked and at not in picked:
                    continue
                out += [(mm, SPOKE, at, None, t) for mm, t in zip(lengths, ids)]
            for at, many, lengths, ids in self.void:
                if not picked or at not in picked:
                    continue
                out += [(mm, CLIP, at, self.gone, t) for mm, t in zip(lengths, ids)]
        self._runs_was, self._runs = key, sorted(out, key=lambda t: -t[0])
        return self._runs

    def _head(self, pen, x, y, wide, name, runs, font):
        # label left, value right, the way the left panel sets its numbers
        about, detail = self.about.get(name), self.detail.get(name, {})
        ends = len({self.name(f) for _, _, _, f, _ in runs})
        rows = []
        # a row is one tract on one connection, so the three numbers differ and
        # each is named for what it is rather than all of them called tracts
        seen, lines = len({t for _, _, _, _, t in runs}), len({(n, f) for _, _, n, f, _ in runs})
        if self.crowd:
            took = sum(self.lost[n] for n in self.crowd)
            rows.append(("parcels behind it", str(len(self.crowd))))
            rows.append(("in the tumor", f"{took:.2f} mL of them"))
            rows.append(("connections", str(lines)))
            rows.append(("tracts", str(seen)))
            rows.append(("rows", str(len(runs))))
            rows.append(("reaching", f"{ends} parcel" + ("s" if ends != 1 else "")))
            pen.text((x, y), name or "", fill=INK, font=font, anchor="lm")
            at = y + 15
            for label, value in rows:
                pen.text((x + 2, at), label, fill=FAINT_INK, font=font, anchor="lm")
                pen.text((x + wide, at), value, fill=INK, font=font, anchor="rm")
                at += 14
            return at + 10
        if about:
            rows.append(("in the tumor", f"{about['in_tumor_ml']:.2f} of "
                                         f"{about['parcel_ml']:.2f} mL"))
            rows.append(("that is", f"{about['fraction']:.0%} of it"))
            rows.append(("mostly", about["tissue"]))
        else:
            rows.append(("in the tumor", "none of it"))
        rows.append(("lobe", detail.get("lobe", "").lower()))
        rows.append(("connections", str(lines)))
        rows.append(("tracts", str(seen)))
        rows.append(("rows", str(len(runs))))
        rows.append(("reaching", f"{ends} parcel" + ("s" if ends != 1 else "")))

        pen.text((x, y), name, fill=INK, font=font, anchor="lm")
        long = detail.get("long_name", "")
        if long:
            pen.text((x, y + 15), long[:44], fill=FAINT_INK, font=font, anchor="lm")
        at = y + (30 if long else 15)
        for label, value in rows:
            pen.text((x + 2, at), label, fill=FAINT_INK, font=font, anchor="lm")
            pen.text((x + wide, at), value, fill=INK, font=font, anchor="rm")
            at += 14
        return at + 10

    def _shelf(self, pen, wide, high):
        # the same dots and lines as the chart, laid out rather than arranged,
        # so that a line's length can be the tract's length. it is about
        # whatever is picked, so with nothing picked there is no panel
        if self.lit is None and self.guest is None and not self.crowd:
            return
        runs = self.runs()
        if wide < self.SHELF + 620:
            return
        name = (self.label if self.crowd else
                self.guest["name"] if self.guest else self.name(self.lit))
        head = 58 + 14 * (7 if self.crowd else 8 if name in self.about else 6)
        # rows keep a readable height and the list scrolls, rather than 534 of
        # them being squeezed into the pane until none of them can be read
        pitch = 15.0
        fits = self.fits = max(int((high - 28 - head - 34) / pitch), 1)
        self.at = int(np.clip(self.at, 0, max(len(runs) - fits, 0)))
        part = runs[self.at:self.at + fits]
        tall = int(head + 34 + len(part) * pitch)
        left, top = wide - 14 - self.SHELF, 14
        pen.rounded_rectangle((left, top, wide - 14, top + tall), radius=6,
                              fill=PANEL, outline=KEY_EDGE)
        font, small = _font(12), _font(11)
        pen.rectangle((left + 10, top - 5, left + 66, top + 5), fill=PANEL)
        pen.text((left + 14, top), "tracts", font=font, anchor="lm", fill=BRIGHT_INK)

        if not runs:
            pen.text((left + 16, top + 24), name or "", fill=INK, font=small, anchor="lm")
            pen.text((left + 16, top + 44), "no tract runs to it",
                     fill=FAINT_INK, font=small, anchor="lm")
            return
        self._head(pen, left + 16, top + 24, self.SHELF - 34, name, runs, small)
        longest = runs[0][0] or 1.0
        runs = part

        cap = max(pitch / 2 - 0.5, 1.0)
        named = pitch >= 12
        # the column is only as wide as the longest name in it, so a panel of
        # short names does not leave the chart squeezed into the right half
        column = (max((pen.textlength(self.name(f) or "", font=small)
                       for _, _, _, f, _ in runs), default=0) + 10) if named else 0
        # the far ends line up in a column on the left with their names beside
        # them, and each tract runs right from there to the parcel that is picked
        edge_x = left + 18 + column + cap
        room = (left + self.SHELF - 18 - cap - (46 if named else 0)) - edge_x
        y0 = top + head
        foot = y0 + len(runs) * pitch

        # the scale runs under the rows, so a length can be read and not guessed
        step = next((v for v in self.STEPS_MM if longest / v <= 5), self.STEPS_MM[-1])
        last = edge_x
        for mm in range(0, int(longest) + step, step):
            at = edge_x + mm / longest * room
            if at > edge_x + room + 2:
                break
            pen.line([(at, y0 - 6), (at, foot + 2)], fill=KEY_EDGE)
            pen.text((at, foot + 12), str(mm), fill=FAINT_INK, font=small, anchor="mm")
            last = at
        pen.text((last + 16, foot + 12), "mm", fill=FAINT_INK, font=small, anchor="lm")

        self._rows_at = (y0, pitch, left + 8, self.SHELF - 16, self.at)
        if len(self._runs) > len(runs):
            # a track down the edge, so a long list says it is a long list
            deep = tall - head - 20
            high_ = max(deep * len(runs) / len(self._runs), 12)
            at = deep * self.at / max(len(self._runs) - len(runs), 1) * (1 - high_ / deep)
            bar = left + self.SHELF - 7
            pen.line([(bar, y0 - 8), (bar, y0 - 8 + deep)], fill=KEY_EDGE)
            pen.line([(bar, y0 - 8 + at), (bar, y0 - 8 + at + high_)], fill=INK, width=2)
        for n, (mm, tone, near, far, tract) in enumerate(runs):
            y = y0 + n * pitch
            if tract == self.tract:
                pen.rectangle([left + 8, y - pitch / 2, left + self.SHELF - 8,
                               y + pitch / 2], fill=(52, 50, 48))
            reach = max(mm / longest * room, 2.0)
            pen.line([(edge_x, y), (edge_x + reach, y)], fill=tone,
                     width=2 if pitch >= 5 else 1)
            for at, side in ((far, edge_x), (near, edge_x + reach)):
                size, fill, edge = self._dot(at)
                size = min(size, cap)
                pen.ellipse([side - size, y - size, side + size, y + size],
                            fill=fill, outline=edge)
            if not named:
                continue
            # the mass is the one yellow dot on the chart, so it names itself
            if self.name(far):
                pen.text((edge_x - cap - 8, y), self.name(far),
                         fill=FAINT_INK, font=small, anchor="rm")
            # the length follows its own bar rather than sitting in a column
            pen.text((edge_x + reach + cap + 5, y), f"{mm:.0f}",
                     fill=FAINT_INK, font=small, anchor="lm")

    def _join(self, pen, a, b, many, lengths, gap, tone, own):
        # one line for one connection. its weight is how many tracts run along
        # it, which is what a plan needs: how much is riding on this link
        start = self.screen[a]
        stop = self.guest["screen"] if b == "guest" else self.screen[b]
        width = (1 if many < 2 else 2 if many < 5 else 3 if many < 12 else 4) + own
        seat_a = (self.screen[a] if a >= len(self.nodes) else self.seat[self.lobes[a]])
        seat_b = (self.seat.get(self.guest["lobe"], self.middle) if b == "guest"
                  else self.screen[b] if b >= len(self.nodes)
                  else self.seat[self.lobes[b]])
        self._ribbon(pen, start, seat_a, stop, seat_b, 1, tone, width)

    def over_shelf(self, x, y):
        if self._rows_at is None:
            return False
        y0, pitch, left, wide, _ = self._rows_at
        return left <= x <= left + wide and y0 - 90 <= y <= y0 + self.fits * pitch

    def scroll_shelf(self, by):
        self.at = max(self.at + by, 0)

    def picked_row(self, x, y):
        # which tract row the pointer is on, if the panel is up and it is on one
        if self._rows_at is None:
            return None
        y0, pitch, left, wide, first = self._rows_at
        runs = self.runs()
        if not (left <= x <= left + wide and y0 - pitch <= y <= y0 + self.fits * pitch):
            return None
        at = first + int(round((y - y0) / max(pitch, 1e-6)))
        return runs[at][4] if 0 <= at < len(runs) else None

    def _neighbors(self):
        # while a node is picked, its own edges stay lit and the rest fall back
        if self.guest is not None:
            return {k for k, *_ in self.guest["link"]}
        picked = self.picked()
        if not picked:
            return None
        near = set(picked)
        for a, b, *_ in self.link:
            if a in picked:
                near.add(b)
            elif b in picked:
                near.add(a)
        return near

    @staticmethod
    def _shade(color, lit):
        return (color if lit else
                tuple(int(c * 0.35 + g * 0.65) for c, g in zip(color, PANEL)))

    def _hot(self):
        # which hubs the pick reaches, so they can show it the way a node does
        out, picked = set(), self.picked()
        if not picked and self.tract is None:
            return out
        for at, many, lengths, ids in self.spokes:
            if at in picked or self.tract in ids:
                out.add(self.hub)
        for at, many, lengths, ids in self.void:
            if at in picked or self.tract in ids:
                out.add(self.gone)
                if at is None:
                    out.add(self.hub)
        return out

    def _nodes(self, pen, near, hot, picked):
        for at, point in enumerate(self.screen):
            last = at >= len(self.nodes)
            if last:
                wide = 11 if at == self.hub else 8
                color = NODE_TUMOR if at == self.hub else STRAY
            else:
                # area says how much of that parcel the tumor took
                wide = 3.0 + 7.0 * (self.lost[at] / self.most) ** 0.5
                color = NODE_IN if self.nodes[at] in self.inside else NODE_OUT
            # a hub is only part of the picture while the pick reaches it
            color = self._shade(color, near is None or
                                (at in hot if last else at in near))
            box = [point[0] - wide, point[1] - wide, point[0] + wide, point[1] + wide]
            hollow = at in self.loose
            ring = at in picked or at in hot
            pen.ellipse(box, fill=PANEL if hollow else color,
                        outline=LIT if ring else color if hollow else PANEL,
                        width=3 if ring else 1)

    def _hubs(self, pen, near, hot):
        # the second one needs saying: it is not a place, it is the absence of
        # one, and both sit under every line in the chart so they need a plate
        font = _font(11)
        for at, text in ((self.hub, "the tumor"), (self.gone, "no parcel")):
            x, y = self.screen[at]
            room = pen.textlength(text, font=font) / 2 + 5
            pen.rectangle([x - room, y + 14, x + room, y + 28], fill=PANEL)
            pen.text((x, y + 21), text, font=font, anchor="mm",
                     fill=self._shade(INK if at == self.hub else FAINT_INK,
                                      near is None or at in hot))

    MANY = 30

    def _labels(self, pen, near):
        if self.zoom < 1.8 and near is None:
            return
        picked = self.picked()
        # one parcel's neighborhood is worth naming; a function's fifty is not,
        # so past a point only the picked ones are named, and then none of them
        show = near
        if near is not None and len(near) > self.MANY:
            show = picked if len(picked) <= self.MANY else set()
        font = _font(11)
        for at, point in enumerate(self.screen[:len(self.nodes)]):
            if show is not None and at not in show and self.zoom < 1.8:
                continue
            pen.text((point[0], point[1] - 10), self.name(at),
                     fill=LIT if at in picked else NODE_OUT, font=font, anchor="ms")

    def _key(self, canvas, pen, wide, high):
        # the same box as every other panel: panel ground, dim round border,
        # a title cut into the top edge, headings over quieter rows
        rows = sorted(self.where)
        tall = 322 + 14 * len(rows)
        if wide < 760 or high < tall + 40:
            return
        if self._key_was != (tuple(rows), tall):
            self._key_was = (tuple(rows), tall)
            self._key_art = self._key_tile(tuple(rows), tall)
        canvas.paste(self._key_art, (14, 6))

    def _key_tile(self, rows, tall):
        # nothing in it moves while the chart is panned, so draw it once
        font, small = _font(12), _font(11)
        # the top edge is inset so the title sitting on it is not clipped
        tile = Image.new("RGB", (620, tall + 9), PANEL)
        pen = Ink(tile)
        pen.rounded_rectangle((0, 8, 619, tall + 8), radius=6, fill=PANEL, outline=KEY_EDGE)
        pen.rectangle((10, 3, 46, 13), fill=PANEL)
        pen.text((14, 8), "key", fill=BRIGHT_INK, font=font, anchor="lm")
        x, y, edge = 14, 34, 606
        count = self.tally
        side = {name: sum(1 for lobe in self.lobes if lobe == name) for name in rows}

        def line(y, text, tally=None, join=None):
            # counts sit in their own columns, the way the left panel sets them
            pen.text((x + 21, y), text, fill=FAINT_INK, font=font, anchor="lm")
            if join is not None:
                pen.text((x + 512, y), str(join), fill=INK, font=font, anchor="rm")
            if tally is not None:
                pen.text((edge, y), str(tally), fill=INK, font=font, anchor="rm")
            return y + 15

        pen.text((x, y), "parcels", fill=INK, font=small, anchor="lm")
        y += 17
        # the two hubs are named on the chart itself, so they are not listed here
        for color, text, tally in (
                (NODE_IN, "the tumor overlaps it", count["in"]),
                (NODE_OUT, "the tumor misses it, but a tract through the tumor "
                           "reaches it", count["out"])):
            pen.ellipse([x + 1, y - 5, x + 11, y + 5], fill=color)
            y = line(y, text, tally)
        pen.ellipse([x + 1, y - 5, x + 11, y + 5], fill=PANEL, outline=NODE_IN)
        y = line(y, "the tumor overlaps it, and no line reaches it",
                 count["loose"]) + 6


        pen.text((x, y), "volume in the tumor", fill=INK, font=small, anchor="lm")
        y += 19
        step = x + 7
        for share in (0.15, 0.5, 1.0):
            size = 3.0 + 7.0 * share ** 0.5
            pen.ellipse([step - size, y - size, step + size, y + size], fill=NODE_IN)
            pen.text((step, y + 16), f"{self.most * share:.1f}", fill=FAINT_INK,
                     font=small, anchor="mm")
            step += 44
        pen.text((step - 8, y + 16), "mL", fill=FAINT_INK, font=small, anchor="lm")
        y += 33

        pen.text((x, y), "lines, one per connection", fill=INK, font=small, anchor="lm")
        pen.text((x + 512, y), "connections", fill=FAINT_INK, font=small, anchor="rm")
        pen.text((edge, y), "tracts", fill=FAINT_INK, font=small, anchor="rm")
        y += 17
        # strongest evidence first, so the last row can refer back to it. the
        # color is about the tract, never about whether the ends are overlapped
        # the column counts lines, so the rows have to be named as lines. they
        # used to say tracts, which is a different object and a different number
        for color, text, key in (
                (THROUGH, "two parcels, joined by tracts that run through the tumor",
                 "pair"),
                (SPOKE, "a parcel and the tumor, by the tracts running through both",
                 "spoke"),
                (STRAY, "the tumor and unnamed tissue, where crossing tracts end",
                 "gone_mass"),
                (CLIP, "a parcel and unnamed tissue, by tracts that miss the tumor",
                 "gone_parcel")):
            pen.line([(x, y), (x + 14, y)], fill=color, width=2)
            y = line(y, text, count[key][1], count[key][0])
        y = line(y, f"{self.crossed} tracts run through or beside the tumor, and")
        y = line(y, "one is counted on each line it runs along")
        y += 9

        pen.text((x, y), "regions, clockwise from the top",
                 fill=INK, font=small, anchor="lm")
        y += 17
        for lobe in rows:
            pen.rectangle([x + 1, y - 5, x + 11, y + 5],
                          fill=_ground(*lobe, lift=1.6), outline=KEY_EDGE)
            pen.text((x + 21, y), f"{lobe[0].lower()} {lobe[1].lower()}",
                     fill=FAINT_INK, font=font, anchor="lm")
            pen.text((edge, y), str(side[lobe]), fill=INK, font=font, anchor="rm")
            y += 14
        return tile

    def pick(self, x, y, reach=14):
        if not len(self.screen):
            return None
        gap = np.linalg.norm(self.screen - np.array([x, y]), axis=1)
        at = int(np.argmin(gap))
        return at if gap[at] <= reach else None


class Slices:
    # Orthogonal planes. Volumes are reoriented to RAS so plane logic is uniform.

    PLANES = ("axial", "coronal", "sagittal")
    MARKS = {"axial": ("R", "L", "A", "P"), "coronal": ("R", "L", "S", "I"),
             "sagittal": ("A", "P", "S", "I")}

    def __init__(self, subject, space="native", later=None, background="1"):
        # subject is the scan plans come from; later, the same patient's follow-up.
        # With both, patient space is the follow-up's grid, where the pre-op scan
        # already sits after its rigid fit, so the two tumors overlay directly
        self.subject, self.later, self.background = subject, later, background
        self.geom = {}
        # per scan: the classes left out, and the fill used when the whole tumor
        # stands in for its parts
        self.hidden, self.whole = {"1": set(), "2": set(LABELS)}, {"1": None, "2": None}
        self.involved, self.lit = {}, None
        # (kind, phase) pairs shown, and the masks loaded for them
        self.targets, self.show_targets = {}, set()
        self.compare, self.pair = {}, None
        self.load(space)

    def _grid(self):
        # whose grid the native planes use
        return self.later if self.later and self.space == "native" else self.subject

    def _seg_path(self, scan):
        who = self.subject if scan == "1" else self.later
        if self.space == "mni":
            return out_dir(who) / "seg_in_mni.nii.gz"
        if scan == "1" and self.later:
            return pair_dir(self.later) / "first_seg_in_later.nii.gz"
        return out_dir(who) / "seg.nii.gz"

    def _first_on_later(self, name):
        # a pre-op sequence moved onto the follow-up's grid by the same rigid fit
        # the tumor took, made once and kept
        # align already made the T1; the others are only ever wanted here
        dest = (pair_dir(self.later) / "first_t1_in_later.nii.gz" if name == "T1"
                else pair_cache(self.later) / f"first_{name}_in_later.nii.gz")
        source = modality(stripped_dir(self.subject), self.subject, name)
        if not dest.exists() and source.exists():
            subprocess.run(["greedy", "-d", "3",
                            "-rf", str(modality(stripped_dir(self.later), self.later, "T1")),
                            "-rm", str(source), str(dest),
                            "-r", str(pair_dir(self.later) / "rigid_first_to_later.mat"),
                            "-threads", THREADS], capture_output=True)
        return dest

    def load(self, space):
        self.space = space
        self.volumes, self.parcels, self.stripped = {}, None, True
        self.compare = {}   # a different grid needs different ones
        self.targets = {}
        shown = self.later if self.background == "2" and self.later else self.subject
        if space == "native":
            if shown != self._grid():
                for m in MODALITIES:
                    p = self._first_on_later(m)
                    if p.exists():
                        self.volumes[m] = self._canon(nib.load(p))
            else:
                # fall back to the raw scans, so a subject mid-pipeline still draws
                d = stripped_dir(shown)
                if not modality(d, shown, "T1").exists():
                    d, self.stripped = raw_dir(shown), False
                for m in MODALITIES:
                    p = modality(d, shown, m)
                    if p.exists():
                        self.volumes[m] = self._canon(nib.load(p))
        else:
            self.volumes["T1"] = self._canon(nib.load(out_dir(shown) / "t1_in_mni.nii.gz"))
            labels, names = atlas.label_volume(nib.load(MNI))
            self.parcels = nib.as_closest_canonical(
                nib.Nifti1Image(labels, nib.load(MNI).affine)).get_fdata().astype(np.int32)
            self.parcel_names = names
        self.affine, first = next(iter(self.volumes.values()))
        self.shape = first.shape
        self.segs = {}
        for scan in ("1", "2") if self.later else ("1",):
            path = self._seg_path(scan)
            if path.exists():
                self.segs[scan] = self._canon(nib.load(path))[1]
        self.segmented = "1" in self.segs
        self.seg = self.segs.get("1", np.zeros(self.shape, dtype=np.int16))
        self.modality = "T1GD" if "T1GD" in self.volumes else next(iter(self.volumes))
        start = (np.unravel_index(np.argmax(self.seg > 0), self.seg.shape)
                 if (self.seg > 0).any() else [s // 2 for s in self.shape])
        self.cursor = [int(v) for v in start]

    @staticmethod
    def _canon(image):
        canonical = nib.as_closest_canonical(image)
        return canonical.affine, np.asarray(canonical.dataobj)

    def _gray(self):
        return self.volumes[self.modality][1]

    def _slab(self, which):
        # radiological: patient left on the viewer's right. RAS x increases right, so flip it.
        x, y, z = self.cursor
        blank = np.zeros_like(self.seg) if self.parcels is None else self.parcels
        planes = []
        later = self.segs.get("2", np.zeros_like(self.seg))
        for volume in (self._gray(), self.seg, blank, later):
            if which == "axial":
                planes.append(volume[::-1, ::-1, z].T)
            elif which == "coronal":
                planes.append(volume[::-1, y, ::-1].T)
            else:
                planes.append(volume[x, ::-1, ::-1].T)
        return planes

    def _marker(self, which):
        # where the crosshair sits inside the slab, in slab pixel coordinates
        x, y, z = self.cursor
        nx, ny, nz = self.shape
        if which == "axial":
            return nx - 1 - x, ny - 1 - y
        if which == "coronal":
            return nx - 1 - x, nz - 1 - z
        return ny - 1 - y, nz - 1 - z

    def canon_mask(self, mask):
        # someone else's array on this space's grid, reoriented the way these are
        name = "seg_in_mni.nii.gz" if self.space == "mni" else "seg.nii.gz"
        reference = nib.load(out_dir(self._grid()) / name)
        return self._canon(nib.Nifti1Image(mask.astype(np.uint8), reference.affine))[1] > 0

    def _target(self, kind, phase):
        # a target on this space's grid, loaded the first time it is shown
        if (kind, phase) not in self.targets:
            source = target_file(self._grid(), kind, phase, self.space)
            self.targets[kind, phase] = (self._canon(nib.load(source))[1] > 0) if source else None
        return self.targets[kind, phase]

    def plane(self, which, size):
        gray, lab, parcel, lab2 = self._slab(which)
        top = np.percentile(gray[gray > 0], 99) if (gray > 0).any() else 1
        rgb = np.repeat((np.clip(gray / max(top, 1) * 255, 0, 255)
                         ).astype(np.uint8)[:, :, None], 3, axis=2)
        # parcels first, so the tumor still wins wherever they overlap
        if self.involved:
            table = np.zeros((len(self.parcel_names) + 1, 3), dtype=np.float32)
            alpha = np.zeros(len(self.parcel_names) + 1, dtype=np.float32)
            for value, color in self.involved.items():
                table[value] = color
                alpha[value] = 0.0 if value == self.lit else 0.22
            weight = alpha[parcel][:, :, None]
            rgb = (rgb * (1 - weight) + table[parcel] * weight).astype(np.uint8)

        # the timepoint comparison, which only mni can line up
        for name, mask in self.compare.items():
            if which == "axial":
                slab = mask[::-1, ::-1, self.cursor[2]].T
            elif which == "coronal":
                slab = mask[::-1, self.cursor[1], ::-1].T
            else:
                slab = mask[self.cursor[0], ::-1, ::-1].T
            if slab.any():
                rgb[slab] = (rgb[slab] * 0.45
                             + np.array(COMPARE[name]) * 0.55).astype(np.uint8)

        # every target as an outline, since a fill would hide what it was grown from
        for kind, phase in sorted(self.show_targets):
            mask = self._target(kind, phase)
            if mask is None:
                continue
            if which == "axial":
                slab = mask[::-1, ::-1, self.cursor[2]].T
            elif which == "coronal":
                slab = mask[::-1, self.cursor[1], ::-1].T
            else:
                slab = mask[self.cursor[0], ::-1, ::-1].T
            ring = outline(slab)
            if ring.any():
                rgb[ring] = np.array(TARGETS[kind][phase], dtype=np.uint8)

        # the pre-op tumor as fills: the whole one under its parts
        if self.whole["1"] is not None:
            every = lab > 0
            if every.any():
                rgb[every] = (rgb[every] * 0.55
                              + np.array(self.whole["1"]) * 0.45).astype(np.uint8)

        for value, color in TISSUE.items():
            if value == 0 or value in self.hidden["1"]:
                continue
            m = lab == value
            if m.any():
                rgb[m] = (rgb[m] * 0.55 + np.array(color) * 0.45).astype(np.uint8)

        # the follow-up as outlines in the same colors, so both can be read at once
        if self.whole["2"] is not None:
            ring = outline(lab2 > 0)
            rgb[ring] = np.array(self.whole["2"], dtype=np.uint8)
        for value, color in TISSUE.items():
            if value == 0 or value in self.hidden["2"]:
                continue
            ring = outline(lab2 == value)
            if ring.any():
                rgb[ring] = np.array(color, dtype=np.uint8)

        # the picked parcel last and solid, so nothing under it tints its color
        if self.lit and self.lit in self.involved:
            on = parcel == self.lit
            if on.any():
                rgb[on] = np.array(self.involved[self.lit], dtype=np.uint8)

        mx, my = self._marker(which)
        if 0 <= my < rgb.shape[0]:
            rgb[my] = (rgb[my] * 0.45 + np.array(CROSS) * 0.55).astype(np.uint8)
        if 0 <= mx < rgb.shape[1]:
            rgb[:, mx] = (rgb[:, mx] * 0.45 + np.array(CROSS) * 0.55).astype(np.uint8)

        im = Image.fromarray(rgb)
        w, h = max(size[0], 8), max(size[1], 8)
        scale = min(w / im.width, h / im.height)
        out = fit(im, (w, h))
        # the click mapping has to know about the letterbox offsets
        self.geom[which] = (scale, im.width, im.height,
                            (w - int(im.width * scale)) // 2, (h - int(im.height * scale)) // 2,
                            w, h)
        return out

    def center(self, parcel):
        if self.parcels is None or parcel not in self.parcel_names:
            return False
        found = np.nonzero(self.parcels == self.parcel_names.index(parcel) + 1)
        if not len(found[0]):
            return False
        # the mean of a curved parcel can fall outside it, so snap to the
        # voxel nearest the mean that is actually in it
        points = np.stack(found, axis=1)
        middle = points.mean(0)
        self.cursor = [int(v) for v in points[np.argmin(((points - middle) ** 2).sum(1))]]
        return True

    def montage(self, cell=700):
        # the three planes side by side, without disturbing the click mapping
        held = dict(self.geom)
        canvas = Image.new("RGB", (cell * len(self.PLANES), cell), PANEL)
        for n, which in enumerate(self.PLANES):
            canvas.paste(self.plane(which, (cell, cell)), (n * cell, 0))
        self.geom = held
        return canvas

    def click(self, which, fx, fy):
        # fx, fy are fractions of the whole pane, letterbox included
        if which not in self.geom:
            return
        scale, iw, ih, ox, oy, bw, bh = self.geom[which]
        px = int((fx * bw - ox) / max(scale, 1e-6))
        py = int((fy * bh - oy) / max(scale, 1e-6))
        if not (0 <= px < iw and 0 <= py < ih):
            return
        nx, ny, nz = self.shape
        if which == "axial":
            self.cursor[0] = max(0, min(nx - 1, nx - 1 - px))
            self.cursor[1] = max(0, min(ny - 1, ny - 1 - py))
        elif which == "coronal":
            self.cursor[0] = max(0, min(nx - 1, nx - 1 - px))
            self.cursor[2] = max(0, min(nz - 1, nz - 1 - py))
        else:
            self.cursor[1] = max(0, min(ny - 1, ny - 1 - px))
            self.cursor[2] = max(0, min(nz - 1, nz - 1 - py))

    def readout(self):
        # only MNI has an anatomical origin; the SRI24 affine is a corner offset,
        # so reporting it as coordinates would be misleading
        x, y, z = self.cursor
        label = int(self.seg[x, y, z])
        later = int(self.segs["2"][x, y, z]) if "2" in self.segs else None
        out = {"voxel": (int(x), int(y), int(z)), "label": LABELS.get(label, "background"),
               "label2": None if later is None else LABELS.get(later, "background"),
               "background": self.background if self.later else "1",
               "label_value": label, "intensity": int(self._gray()[x, y, z]),
               "space": self.space, "modality": self.modality, "stripped": self.stripped,
               "segmented": self.segmented,
               "modalities": list(self.volumes),
               "signal": {m: int(v[1][x, y, z]) for m, v in self.volumes.items()},
               "parcel": None}
        if self.parcels is not None:
            value = int(self.parcels[x, y, z])
            if 0 < value <= len(self.parcel_names):
                out["parcel"] = self.parcel_names[value - 1]
        if self.space == "mni":
            world = nib.affines.apply_affine(self.affine, [x, y, z])
            out["coords"] = tuple(round(float(v)) for v in world)
            out["coord_space"] = "MNI mm"
        else:
            out["coords"] = None
            out["coord_space"] = "follow-up grid" if self.later else "SRI24 grid"
        return out
