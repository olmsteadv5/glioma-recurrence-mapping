# Offscreen rendering for the terminal UI: 3D scenes and slice planes, PIL images out.

import functools
import os

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkWindowToImageFilter
from PIL import Image, ImageDraw, ImageFilter

import atlas
import tracts as tractography
import view
from paths import LABELS, MNI, MODALITIES, modality, out_dir, raw_dir, stripped_dir

pv.OFF_SCREEN = True

BG = "#282828"
SHELL = (168, 153, 132)
TRACT = (95, 155, 114)
TRACT_MASS = (198, 124, 72)
TRACT_NEAR = (118, 160, 150)
TISSUE = {1: (242, 193, 78), 2: (63, 167, 214), 3: (228, 87, 46), 4: (142, 125, 190),
          0: (214, 200, 175)}  # 0 is the three of them as one shape
CROSS = (250, 189, 47)
LIT = (251, 241, 199)
WHOLE = "WT"
MARGIN = (177, 98, 134)
COMPARE = {"both": (235, 219, 178), "new": (251, 73, 52), "gone": (142, 192, 124)}
MARGIN_MM = 20
_hex = lambda t: "#%02x%02x%02x" % tuple(t)
PANEL = (40, 40, 40)


def fit(image, size, background=PANEL):
    # Always hand back exactly the requested box, so the widget never shows dead space.
    w, h = max(size[0], 8), max(size[1], 8)
    scale = min(w / image.width, h / image.height)
    scaled = image.resize((max(int(image.width * scale), 1), max(int(image.height * scale), 1)),
                          Image.NEAREST)
    canvas = Image.new("RGB", (w, h), background)
    canvas.paste(scaled, ((w - scaled.width) // 2, (h - scaled.height) // 2))
    return canvas


def grow(seg_image, brain, mm=MARGIN_MM):
    # the planning margin, clipped to the brain because a fifth of a raw
    # expansion lands in skull and air
    far = ndimage.distance_transform_edt(np.asarray(seg_image.dataobj) == 0,
                                         sampling=seg_image.header.get_zooms()[:3])
    return (far <= mm) & brain


def outline(mask):
    # the boundary only, so a 2 cm envelope does not swamp the slice
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


class Scene:
    # One plotter per subject; between frames only the camera moves.

    def __init__(self, subject, overlap, palette="contrast", size=(700, 540)):
        self.subject, self.overlap, self.palette = subject, overlap, palette
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

        image = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
        for value, (name, points, faces) in self._tumour(subject, image).items():
            self.actors["T:" + name] = self._mesh(points, faces, _hex(TISSUE[value]),
                                                  0.5 if value == 0 else 1.0)

        self.tract_counts = {}
        for key, color, lines in self._tracts(image, index, involved):
            self.tract_counts[key] = len(lines)
            mesh = self._lines(lines)
            if mesh is not None:
                self.actors[key] = self.plotter.add_mesh(
                    mesh, color=_hex(color), line_width=1, opacity=0.55)

        self._picker, self.lit = None, None
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
    def _tumour(subject, image):
        # contouring the four labels is half a second, and it only changes
        # when the registration does
        cache = out_dir(subject) / "tumour_surfaces.npz"
        if cache.exists() and cache.stat().st_mtime >= os.path.getmtime(image.get_filename()):
            held = np.load(cache)
            return {int(v): (WHOLE if int(v) == 0 else LABELS[int(v)],
                             held[f"v{v}"], held[f"f{v}"]) for v in held["values"]}
        data = np.asarray(image.dataobj).astype(np.int16)
        out, saved = {}, {"values": []}
        # value 0 is all three classes contoured as one closed surface
        for value, name in list(LABELS.items()) + [(0, WHOLE)]:
            source = (data > 0).astype(np.int16) if value == 0 else data
            surface = view._tumour_surface(source, image.affine, 1 if value == 0 else value)
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

    def _tracts(self, seg_image, index, involved):
        # the whole set, then the two ways a line earns its place in it
        labels, names = atlas.label_volume(nib.load(MNI))
        wanted = [index[n] + 1 for n in involved if n in index]
        mass, nearby = tractography.crossing(tractography.centrelines(),
                                             seg_image, labels, wanted)
        return [("tracts", TRACT, mass + nearby), ("tracts:mass", TRACT_MASS, mass),
                ("tracts:near", TRACT_NEAR, nearby)]

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

    def ensure_margin(self):
        # only built when it is asked for: the distance transform is not free
        if "T:margin" in self.actors:
            return
        cache = out_dir(self.subject) / "margin_surface.npz"
        if cache.exists():
            held = np.load(cache)
            points, faces = held["vertices"], held["faces"]
        else:
            image = nib.load(out_dir(self.subject) / "seg_in_mni.nii.gz")
            grown = grow(image, np.asarray(nib.load(MNI).dataobj) > 0)
            points, faces = view._tumour_surface(grown.astype(np.int16), image.affine, 1)
            np.savez_compressed(cache, vertices=points, faces=faces)
        self.actors["T:margin"] = self._mesh(points, faces, _hex(MARGIN), 0.25)

    def ensure_compare(self, first, second):
        # the two timepoints against each other, built the first time it is asked for
        if "C:both" in self.actors:
            return
        import compare
        for name, (points, faces) in compare.surfaces(first, second).items():
            self.actors["C:" + name] = self._mesh(points, faces, _hex(COMPARE[name]), 0.6)

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


@functools.lru_cache(maxsize=1)
def _bows():
    # How far to bend an arc so that it comes out a given multiple of its chord.
    steps = np.linspace(0, 1, 32)[:, None]
    heights = np.linspace(0, 3.2, 129)
    grown = []
    for height in heights:
        curve = ((1 - steps) ** 2 * np.array([0.0, 0.0])
                 + 2 * (1 - steps) * steps * np.array([0.5, height])
                 + steps ** 2 * np.array([1.0, 0.0]))
        grown.append(float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()))
    return np.array(grown), heights


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


def bow_for(ratio):
    grown, heights = _bows()
    return float(np.interp(min(max(ratio, 1.0), grown[-1]), grown, heights))


NODE_IN = (228, 87, 46)
NODE_OUT = (131, 165, 152)
NODE_TUMOUR = (250, 189, 47)
# one hue per kind of link: cool where the mass is not in the path, warm where
# it is. strength brightens a line without draining its hue
EDGE = (84, 102, 112)
SPOKE = (152, 100, 108)
THROUGH = (154, 112, 72)
# the interface palette, so the key is the same greys as every other panel
INK = (168, 153, 132)
FAINT_INK = (124, 111, 100)
BRIGHT_INK = (235, 219, 178)
KEY_EDGE = (80, 73, 69)
# one soft ground per hemisphere and lobe, dark enough to sit under the chart.
# the two sides of a lobe land far apart, so one ground for both would stretch
# across half the picture and say nothing
LOBE = {"Frontal": (64, 82, 104), "Parietal": (56, 92, 82), "Temporal": (100, 72, 58),
        "Occipital": (82, 62, 96), "Insula": (102, 84, 52),
        "Corpus callosum": (78, 78, 78)}


def _ground(side, lobe, lift=0.55):
    base = LOBE.get(lobe, (78, 78, 78))
    dim = 0.6 if side == "Right" else 1.0
    return tuple(int(c * dim * lift) for c in base)


class Chart:
    # The tumour's neighbourhood as a node chart, laid out so the gaps between
    # nodes are the real distances between the parcels.

    def __init__(self, subject, overlap):
        import graph
        self.subject = subject
        (self.nodes, self.link, to_tumour,
         self.inside, self.spokes) = graph.build(subject, overlap)
        self.points = graph.layout(len(self.nodes), self.link, to_tumour)
        info = atlas.parcels()
        self.names = [p["name"] for p in info]
        self.lobes = [(info[n]["hemisphere"], info[n]["lobe"]) for n in self.nodes]

        lost = {r["parcel"]: r["in_tumour_ml"] for r in overlap["parcels"]}
        self.lost = [lost.get(self.names[n], 0.0) for n in self.nodes]
        self.most = max(self.lost, default=1.0) or 1.0
        # no edge and no spoke reaches these, so they are drawn hollow rather
        # than left looking like the layout dropped them
        self.loose = (set(range(len(self.nodes)))
                      - {e[0] for e in self.link} - {e[1] for e in self.link}
                      - {s[0] for s in self.spokes})

        # two thirds of the parcel to parcel lines stand for a strength with no
        # tract behind it, so the count is of lines drawn, not tracts measured
        self.strands = (sum(len(r) for _, _, r in self.spokes)
                        + sum(len(e[4]) for e in self.link))
        bare = sum(len(e[4]) for e in self.link if e[4] == [1.0])
        self.tally = {
            "in": len(self.inside), "out": len(self.nodes) - len(self.inside),
            "mass": 1, "loose": len(self.loose),
            "plain": sum(len(e[4]) for e in self.link if not e[5]),
            "via": sum(len(e[4]) for e in self.link if e[5]),
            "spoke": sum(len(r) for _, _, r in self.spokes),
            "bow": self.strands - bare, "bare": bare}

        self.lit, self.guest, self.zoom, self.pan = None, None, 1.0, np.zeros(2)
        self._key_was, self._key_art = None, None
        self.where = {}
        self.screen, self.scale = np.zeros((0, 2)), 1.0

    def name(self, at):
        if at is None or at >= len(self.nodes):
            return None
        return self.names[self.nodes[at]]

    def focus(self, parcel):
        # the node for the picked row: the one already drawn, or a guest placed
        # by its real distances, since three quarters of the connectivity rows
        # name a parcel too weakly tied to have made the chart
        import graph
        self.lit = self.guest = None
        if parcel is None:
            return
        self.lit = next((n for n in range(len(self.nodes))
                         if self.names[self.nodes[n]] == parcel), None)
        if self.lit is not None:
            return
        made = graph.attach(self.subject, self.nodes, parcel)
        if made is None:
            return
        n, link, far = made
        info = atlas.parcels()[n]
        self.guest = {"name": parcel, "link": link,
                      "lobe": (info["hemisphere"], info["lobe"]),
                      "point": graph.settle(self.points, link, far),
                      "screen": np.zeros(2)}

    def move(self, zoom=1.0, pan=(0.0, 0.0)):
        self.zoom = float(np.clip(self.zoom * zoom, 0.4, 12.0))
        self.pan = self.pan + np.array(pan, dtype=float)

    def home(self):
        self.zoom, self.pan = 1.0, np.zeros(2)

    def _place(self, wide, high):
        pad = 30
        low, span = self.points.min(0), np.ptp(self.points, axis=0)
        span[span == 0] = 1
        self.scale = float(min((wide - 2 * pad) / span[0], (high - 2 * pad) / span[1]) * self.zoom)
        self._shift = -low - span / 2
        self._origin = np.array([wide, high]) / 2 + self.pan
        self.screen = (self.points + self._shift) * self.scale + self._origin
        if self.guest is not None:
            self.guest["screen"] = ((self.guest["point"] + self._shift) * self.scale
                                    + self._origin)

    def image(self, size):
        wide, high = max(size[0], 64), max(size[1], 64)
        canvas = Image.new("RGB", (wide, high), PANEL)
        if not len(self.points):
            return canvas
        pen = ImageDraw.Draw(canvas)
        self._place(wide, high)
        near = self._neighbours()

        self._blobs(canvas, wide, high)
        # a line stays lit only if the picked node is one of its two ends. the
        # neighbours themselves stay lit, but the lines between them do not,
        # which used to read as the picked parcel reaching further than it does
        free = near is None
        for at, count, ratios in self.spokes:
            own = at == self.lit
            self._strands(pen, self.screen[at], self.screen[-1], ratios,
                          self._shade(SPOKE, free or own), 3 if own else 2)
        for a, b, weight, _, ratios, crosses in self.link:
            own = self.lit in (a, b)
            self._strands(pen, self.screen[a], self.screen[b], ratios,
                          self._shade(self._tone(weight, crosses), free or own),
                          3 if own else 2)
        if self.guest is not None:
            for k, weight, _, ratios, crosses in self.guest["link"]:
                self._strands(pen, self.screen[k], self.guest["screen"], ratios,
                              self._tone(weight, crosses), 3)
        self._nodes(pen, near)
        if self.guest is not None:
            self._guest(pen)
        self._labels(pen, near)
        self._key(canvas, pen, wide, high)
        return canvas

    @staticmethod
    def _tone(weight, crosses):
        base = THROUGH if crosses else EDGE
        gain = 1.0 + 0.9 * min(weight, 1.0)
        return tuple(min(int(c * gain), 240) for c in base)

    def _guest(self, pen):
        # nothing of it is gone, so it draws at the smallest size the key shows
        point, wide = self.guest["screen"], 3.0
        pen.ellipse([point[0] - wide, point[1] - wide, point[0] + wide, point[1] + wide],
                    fill=NODE_OUT, outline=LIT, width=3)
        pen.text((point[0], point[1] - 14), self.guest["name"], fill=LIT,
                 font=_font(11), anchor="ms")

    def _neighbours(self):
        # while a node is picked, its own edges stay lit and the rest fall back
        if self.guest is not None:
            return {k for k, *_ in self.guest["link"]}
        if self.lit is None:
            return None
        near = {self.lit}
        for a, b, *_ in self.link:
            if a == self.lit:
                near.add(b)
            elif b == self.lit:
                near.add(a)
        return near

    @staticmethod
    def _shade(color, lit):
        return color if lit else tuple(int(c * 0.35 + 40 * 0.65) for c in color)

    def _blobs(self, canvas, wide, high):
        # a soft ground under each lobe's parcels, built small and blurred up
        step = 4
        small = (max(wide // step, 8), max(high // step, 8))
        reach = float(max(self.scale * 7.5 / step, 2.5))
        self.where = {}
        for lobe in sorted(set(self.lobes)):
            picked = [i for i, name in enumerate(self.lobes) if name == lobe]
            if len(picked) < 3:
                continue
            mask = Image.new("L", small, 0)
            brush = ImageDraw.Draw(mask)
            for i in picked:
                x, y = self.screen[i] / step
                brush.ellipse([x - reach, y - reach, x + reach, y + reach], fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(reach * 0.55))
            mask = mask.point(lambda v: 255 if v > 110 else 0)
            mask = mask.filter(ImageFilter.GaussianBlur(reach * 0.3))
            # the word belongs on the blob, not at the middle of scattered nodes
            grid = np.asarray(mask, dtype=np.float32)
            if grid.sum() > 0:
                ys, xs = np.nonzero(grid)
                self.where[lobe] = (float(xs.mean()) * step, float(ys.mean()) * step)
            mask = mask.resize((wide, high), Image.BILINEAR).point(lambda v: int(v * 0.42))
            canvas.paste(Image.new("RGB", (wide, high), _ground(*lobe)), (0, 0), mask)



    def _nodes(self, pen, near):
        for at, point in enumerate(self.screen):
            last = at == len(self.screen) - 1
            if last:
                wide = 11
                color = NODE_TUMOUR
            else:
                # area says how much of that parcel the tumour took
                wide = 3.0 + 7.0 * (self.lost[at] / self.most) ** 0.5
                color = NODE_IN if self.nodes[at] in self.inside else NODE_OUT
            color = self._shade(color, near is None or at in near)
            box = [point[0] - wide, point[1] - wide, point[0] + wide, point[1] + wide]
            hollow = at in self.loose
            pen.ellipse(box, fill=PANEL if hollow else color,
                        outline=(LIT if at == self.lit
                                 else color if hollow else PANEL),
                        width=3 if at == self.lit else 1)

    def _labels(self, pen, near):
        if self.zoom < 1.8 and near is None:
            return
        font = _font(11)
        for at, point in enumerate(self.screen[:-1]):
            if near is not None and at not in near and self.zoom < 1.8:
                continue
            pen.text((point[0], point[1] - 10), self.name(at),
                     fill=LIT if at == self.lit else NODE_OUT, font=font, anchor="ms")

    def _key(self, canvas, pen, wide, high):
        # the same box as every other panel: panel ground, dim round border,
        # a title cut into the top edge, headings over quieter rows
        font = _font(12)
        rows = sorted(self.where)
        tall = 262 + 14 * len(rows)
        if wide < 580 or high < tall + 40:
            self._scale_bar(pen, wide, high, font)
            return
        if self._key_was != (tuple(rows), tall):
            self._key_was = (tuple(rows), tall)
            self._key_art = self._key_tile(tuple(rows), tall)
        canvas.paste(self._key_art, (14, 6))
        self._scale_bar(pen, wide, high, font)

    def _key_tile(self, rows, tall):
        # nothing in it moves while the chart is panned, so draw it once
        font, small = _font(12), _font(11)
        # the top edge is inset so the title sitting on it is not clipped
        tile = Image.new("RGB", (440, tall + 9), PANEL)
        pen = ImageDraw.Draw(tile)
        pen.rounded_rectangle((0, 8, 439, tall + 8), radius=6, fill=PANEL, outline=KEY_EDGE)
        pen.rectangle((10, 3, 46, 13), fill=PANEL)
        pen.text((14, 8), "key", fill=BRIGHT_INK, font=font, anchor="lm")
        x, y, edge = 14, 34, 426
        count = self.tally
        side = {name: sum(1 for lobe in self.lobes if lobe == name) for name in rows}

        def line(y, text, tally=None):
            # the count sits in its own column, the way the left panel sets them
            pen.text((x + 21, y), text, fill=FAINT_INK, font=font, anchor="lm")
            if tally is not None:
                pen.text((edge, y), str(tally), fill=INK, font=font, anchor="rm")
            return y + 15

        pen.text((x, y), "parcels", fill=INK, font=small, anchor="lm")
        y += 17
        for color, text, tally in ((NODE_IN, "overlapped by the tumour", count["in"]),
                                   (NODE_OUT, "not overlapped, linked to an overlapped parcel",
                                    count["out"]),
                                   (NODE_TUMOUR, "the tumour mass", count["mass"])):
            pen.ellipse([x + 1, y - 5, x + 11, y + 5], fill=color)
            y = line(y, text, tally)
        pen.ellipse([x + 1, y - 5, x + 11, y + 5], fill=PANEL, outline=NODE_IN)
        y = line(y, "overlapped, with nothing linking to it", count["loose"]) + 6

        pen.text((x, y), "volume in the tumour", fill=INK, font=small, anchor="lm")
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

        pen.text((x, y), "links", fill=INK, font=small, anchor="lm")
        y += 17
        for color, text, tally in (
                (self._tone(0.5, False), "parcel to parcel", count["plain"]),
                (self._tone(0.5, True), "parcel to parcel, through the mass", count["via"]),
                (SPOKE, "parcel to the mass", count["spoke"])):
            pen.line([(x, y), (x + 14, y)], fill=color, width=2)
            y = line(y, text, tally)
        y += 9

        pen.text((x, y), "regions", fill=INK, font=small, anchor="lm")
        y += 17
        for lobe in rows:
            pen.rectangle([x + 1, y - 5, x + 11, y + 5],
                          fill=_ground(*lobe, lift=1.6), outline=KEY_EDGE)
            pen.text((x + 21, y), f"{lobe[0].lower()} {lobe[1].lower()}",
                     fill=FAINT_INK, font=font, anchor="lm")
            pen.text((edge, y), str(side[lobe]), fill=INK, font=font, anchor="rm")
            y += 14
        return tile

    def _scale_bar(self, pen, wide, high, font):
        bar = 50.0 * self.scale
        if not 20 < bar < wide - 60:
            return
        y = high - 18
        pen.line([(20, y), (20 + bar, y)], fill=INK, width=1)
        for end in (20, 20 + bar):
            pen.line([(end, y - 3), (end, y + 3)], fill=INK)
        pen.text((34 + bar, y), "50 mm between centres", fill=INK, font=font, anchor="lm")

    def _strands(self, pen, start, stop, ratios, color, width=2):
        # one arc per tract, bowed until its drawn length is its real length
        along = stop - start
        span = float(np.linalg.norm(along))
        if span < 1:
            return
        side = np.array([-along[1], along[0]]) / span
        steps = np.linspace(0, 1, 16)[:, None]
        for n, ratio in enumerate(ratios):
            bow = span * bow_for(min(ratio, 1.8)) * (1 if n % 2 else -1)
            middle = (start + stop) / 2 + side * bow
            curve = ((1 - steps) ** 2 * start + 2 * (1 - steps) * steps * middle
                     + steps ** 2 * stop)
            pen.line([tuple(point) for point in curve], fill=color, width=width)

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

    def __init__(self, subject, space="native"):
        self.subject = subject
        self.geom, self.hidden = {}, set()
        self.involved, self.lit, self.whole = {}, None, None
        self.margin, self.show_margin = None, False
        self.compare, self.pair = {}, None
        self.load(space)

    def load(self, space):
        self.space = space
        self.volumes, self.parcels, self.stripped = {}, None, True
        self.margin, self.compare = None, {}   # a different grid needs different ones
        if space == "native":
            # fall back to the raw scans, so a subject mid-pipeline still draws
            d = stripped_dir(self.subject)
            if not modality(d, self.subject, "T1").exists():
                d, self.stripped = raw_dir(self.subject), False
            for m in MODALITIES:
                p = modality(d, self.subject, m)
                if p.exists():
                    self.volumes[m] = self._canon(nib.load(p))
            seg = out_dir(self.subject) / "seg.nii.gz"
        else:
            self.volumes["T1"] = self._canon(nib.load(out_dir(self.subject) / "t1_in_mni.nii.gz"))
            seg = out_dir(self.subject) / "seg_in_mni.nii.gz"
            labels, names = atlas.label_volume(nib.load(MNI))
            self.parcels = nib.as_closest_canonical(
                nib.Nifti1Image(labels, nib.load(MNI).affine)).get_fdata().astype(np.int32)
            self.parcel_names = names
        self.affine, first = next(iter(self.volumes.values()))
        self.shape = first.shape
        self.segmented = seg.exists()
        self.seg = (self._canon(nib.load(seg))[1] if self.segmented
                    else np.zeros(self.shape, dtype=np.int16))
        self.modality = "T1GD" if "T1GD" in self.volumes else next(iter(self.volumes))
        start = (np.unravel_index(np.argmax(self.seg > 0), self.seg.shape)
                 if (self.seg > 0).any() else [s // 2 for s in self.shape])
        self.cursor = [int(v) for v in start]

    @staticmethod
    def _canon(image):
        canonical = nib.as_closest_canonical(image)
        return canonical.affine, np.asarray(canonical.dataobj)

    def _grey(self):
        return self.volumes[self.modality][1]

    def _slab(self, which):
        # radiological: patient left on the viewer's right. RAS x increases right, so flip it.
        x, y, z = self.cursor
        blank = np.zeros_like(self.seg) if self.parcels is None else self.parcels
        planes = []
        for volume in (self._grey(), self.seg, blank, self._margin()):
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
        reference = nib.load(out_dir(self.subject) / name)
        return self._canon(nib.Nifti1Image(mask.astype(np.uint8), reference.affine))[1] > 0

    def _margin(self):
        if self.show_margin and self.margin is None and self.segmented:
            path = out_dir(self.subject) / ("seg_in_mni.nii.gz" if self.space == "mni"
                                            else "seg.nii.gz")
            brain = self._grey() > 0
            self.margin = self._canon(nib.Nifti1Image(
                grow(nib.load(path), np.ones_like(brain)).astype(np.uint8),
                nib.load(path).affine))[1] & (next(iter(self.volumes.values()))[1] > 0)
        return self.margin if self.margin is not None else np.zeros_like(self.seg)

    def plane(self, which, size):
        grey, lab, parcel, edge = self._slab(which)
        top = np.percentile(grey[grey > 0], 99) if (grey > 0).any() else 1
        rgb = np.repeat((np.clip(grey / max(top, 1) * 255, 0, 255)
                         ).astype(np.uint8)[:, :, None], 3, axis=2)
        # parcels first, so the tumour still wins wherever they overlap
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

        if self.show_margin:
            ring = outline(edge > 0)
            if ring.any():
                rgb[ring] = np.array(MARGIN, dtype=np.uint8)

        # the whole tumour as one fill, under its three parts
        if self.whole is not None:
            every = lab > 0
            if every.any():
                rgb[every] = (rgb[every] * 0.55
                              + np.array(self.whole) * 0.45).astype(np.uint8)

        for value, color in TISSUE.items():
            if value == 0 or value in self.hidden:
                continue
            m = lab == value
            if m.any():
                rgb[m] = (rgb[m] * 0.55 + np.array(color) * 0.45).astype(np.uint8)

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

    def centre(self, parcel):
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
        out = {"voxel": (int(x), int(y), int(z)), "label": LABELS.get(label, "background"),
               "label_value": label, "intensity": int(self._grey()[x, y, z]),
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
            out["coord_space"] = "SRI24 grid"
        return out
