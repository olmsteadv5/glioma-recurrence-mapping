
import os
import subprocess

import nibabel as nib
import numpy as np
import pyvista as pv
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
from PIL import Image

from show.compare import _inverse, tumor_surface
from yale import atlas
from hcp import streamlines
from geometry import outline
from paths import (LABELS, MODALITIES, THREADS, anatomy, cache_dir, modality, out_dir, pair_cache,
                   pair_dir, raw_dir, stripped_dir, walk_dir)

pv.OFF_SCREEN = True



def _panel(base=(40, 40, 40)):
    # explainer: kitty draws cell backgrounds at background_opacity but images opaque, so paint what the panels composite to
    where = os.environ.get("KITTY_CONFIG_DIRECTORY", os.path.expanduser("~/.config/kitty"))
    opacity = 1.0
    try:
        for line in open(os.path.join(where, "kitty.conf")):
            if line.split()[:1] == ["background_opacity"]:
                opacity = min(max(float(line.split()[1]), 0.0), 1.0)
    except (OSError, ValueError, IndexError):
        pass
    # explainer: compositor behind is black, so it's a plain scale
    return tuple(round(c * opacity) for c in base)


PANEL = _panel()
BG = "#%02x%02x%02x" % PANEL
SHELL = (168, 153, 132)
TRACTS = {"1": (198, 124, 72), "2": (211, 134, 155), "both": (235, 219, 178)}
TISSUE = {1: (242, 193, 78), 2: (63, 167, 214), 3: (228, 87, 46), 4: (142, 125, 190),
          0: (214, 200, 175)}
CROSS = (250, 189, 47)
WHOLE = "WT"
COMPARE = {"both": (235, 219, 178), "new": (251, 73, 52), "gone": (142, 192, 124)}
WALK = {"46": (131, 165, 152), "60": (215, 153, 33)}
NRG = {"46": (177, 98, 134), "60": (126, 108, 204)}
WALK_PHASES = tuple(WALK)
TARGETS = {"walk": WALK, "nrg": NRG}
TARGET_KEY = {"walk": "W:", "nrg": "T:"}


def target_file(subject, kind, phase, space):
    # explainer: both scans show the one pre-op target. None if walkdown.py hasn't run
    mni = walk_dir(subject) / f"{kind}_{phase}_mni.nii.gz"
    if not mni.exists():
        return None
    if space == "mni":
        return mni
    if subject.endswith("_21"):
        here = walk_dir(subject) / f"{kind}_{phase}.nii.gz"
        return here if here.exists() else None
    here = cache_dir(subject) / f"{kind}_{phase}.nii.gz"
    if not here.exists() or here.stat().st_mtime < mni.stat().st_mtime:
        # explainer: back to the pre-op grid through its own inverse
        subprocess.run(["greedy", "-d", "3", "-rf", str(anatomy(stripped_dir(subject), subject)),
                        "-ri", "LABEL", "0.2vox", "-rm", str(mni), str(here),
                        "-r", str(out_dir(subject) / "affine.mat") + ",-1", str(_inverse(subject)),
                        "-threads", THREADS], capture_output=True)
    return here if here.exists() else None


def walk_ends(subject):
    out = {}
    for phase in WALK_PHASES:
        path = walk_dir(subject) / f"walk_{phase}_ends.npy"
        if path.exists():
            out[phase] = np.load(path)
    return out
_hex = lambda t: "#%02x%02x%02x" % tuple(t)


def fit(image, size, background=PANEL):
    # decision: always return the exact box asked for, so the widget never shows dead space
    w, h = max(size[0], 8), max(size[1], 8)
    scale = min(w / image.width, h / image.height)
    scaled = image.resize((max(int(image.width * scale), 1), max(int(image.height * scale), 1)),
                          Image.NEAREST)
    canvas = Image.new("RGB", (w, h), background)
    canvas.paste(scaled, ((w - scaled.width) // 2, (h - scaled.height) // 2))
    return canvas


class Scene:
    # decision: one plotter per subject, only the camera moves between frames

    def __init__(self, subject, size=(700, 540), later=None):
        self.subject, self.later = subject, later
        self.plotter = pv.Plotter(off_screen=True, window_size=list(size))
        self.plotter.set_background(BG)
        self.actors = {}

        self.actors["shell"] = self._mesh(*atlas.brain(), _hex(SHELL), 0.15)

        # decision: follow-up is see-through so the pre-op tumor shows under it
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

        self.plotter.show(auto_close=False)

        # decision: nobody watches this window, so skip vsync and read the back buffer, much faster
        window = self.plotter.ren_win
        window.SetSwapBuffers(0)
        # explainer: VTK only resolves the back buffer when multisampling is off
        window.SetMultiSamples(0)
        self.grab = vtkWindowToImageFilter()
        self.grab.SetInput(window)
        self.grab.FixBoundaryOn()
        self.grab.ShouldRerenderOff()
        self.grab.SetInputBufferTypeToRGB()
        self.grab.ReadFrontBufferOff()

    @staticmethod
    def _tumor(subject, image):
        # decision: contouring is slow, cache until the registration changes
        cache = cache_dir(subject) / "tumor_surfaces.npz"
        if cache.exists() and cache.stat().st_mtime >= os.path.getmtime(image.get_filename()):
            held = np.load(cache)
            return {int(v): (WHOLE if int(v) == 0 else LABELS[int(v)],
                             held[f"v{v}"], held[f"f{v}"]) for v in held["values"]}
        data = np.asarray(image.dataobj).astype(np.int16)
        out, saved = {}, {"values": []}
        for value, name in list(LABELS.items()) + [(0, WHOLE)]:
            source = (data > 0).astype(np.int16) if value == 0 else data
            surface = tumor_surface(source, image.affine, 1 if value == 0 else value)
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
        # decision: precompute which tracts hit each scan and class, so toggling tissue doesn't redo the search
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
        # decision: a streamline through both scans is drawn once in its own color
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
                    mesh, color=_hex(TRACTS[group]), line_width=1, opacity=0.55)

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
        # explainer: camera.azimuth doesn't actually rotate, these VTK calls do
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

    def ensure_target(self, kind, phase):
        # decision: cached on the mask's mtime so a rebuilt target never shows the old shape
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
            surface = tumor_surface(np.asarray(image.dataobj).astype(np.int16), image.affine, 1)
            if surface is None:
                return
            points, faces = surface
            np.savez_compressed(cache, vertices=points, faces=faces)
        self.actors[key] = self._mesh(points, faces, _hex(TARGETS[kind][phase]), 0.22)

    def ensure_ends(self, phase):
        key = "W:ends" + phase
        if key in self.actors:
            return
        ends = walk_ends(self.subject).get(phase)
        if ends is None or not len(ends):
            return
        self.actors[key] = self.plotter.add_mesh(pv.PolyData(ends), color=_hex(WALK[phase]),
                                                 point_size=4, render_points_as_spheres=True)

    def ensure_compare(self, first, second):
        if "C:both" in self.actors:
            return
        from show import compare
        for name, (points, faces) in compare.surfaces(first, second).items():
            self.actors["C:" + name] = self._mesh(points, faces, _hex(COMPARE[name]), 0.6)

    def image(self):
        self.plotter.render()
        self.grab.Modified()
        self.grab.Update()
        data = self.grab.GetOutput()
        width, height, _ = data.GetDimensions()
        pixels = vtk_to_numpy(data.GetPointData().GetScalars())
        return Image.fromarray(pixels.reshape(height, width, 3)[::-1])


class Slices:
    # decision: reorient to RAS so the plane logic is the same everywhere

    PLANES = ("axial", "coronal", "sagittal")
    MARKS = {"axial": ("R", "L", "A", "P"), "coronal": ("R", "L", "S", "I"),
             "sagittal": ("A", "P", "S", "I")}

    def __init__(self, subject, space="native", later=None, background="1"):
        # explainer: with both scans, native means the follow-up's grid, the pre-op scan is already rigidly on it
        self.subject, self.later, self.background = subject, later, background
        self.geom = {}
        self.hidden, self.whole = {"1": set(), "2": set(LABELS)}, {"1": None, "2": None}
        self.targets, self.show_targets = {}, set()
        self.compare, self.pair = {}, None
        self.load(space)

    def _grid(self):
        return self.later if self.later and self.space == "native" else self.subject

    def _seg_path(self, scan):
        who = self.subject if scan == "1" else self.later
        if self.space == "mni":
            return out_dir(who) / "seg_in_mni.nii.gz"
        if scan == "1" and self.later:
            return pair_dir(self.later) / "first_seg_in_later.nii.gz"
        return out_dir(who) / "seg.nii.gz"

    def _first_on_later(self, name):
        # explainer: pre-op sequence moved onto the follow-up grid by the same rigid fit as the tumor
        dest = (pair_dir(self.later) / "first_t1_in_later.nii.gz" if name == "T1"
                else pair_cache(self.later) / f"first_{name}_in_later.nii.gz")
        source = modality(stripped_dir(self.subject), self.subject, name)
        if not dest.exists() and source.exists():
            subprocess.run(["greedy", "-d", "3",
                            "-rf", str(anatomy(stripped_dir(self.later), self.later)),
                            "-rm", str(source), str(dest),
                            "-r", str(pair_dir(self.later) / "rigid_first_to_later.mat"),
                            "-threads", THREADS], capture_output=True)
        return dest

    def load(self, space):
        self.space = space
        self.volumes, self.stripped = {}, True
        self.compare = {}
        self.targets = {}
        shown = self.later if self.background == "2" and self.later else self.subject
        if space == "native":
            if shown != self._grid():
                for m in MODALITIES:
                    p = self._first_on_later(m)
                    if p.exists():
                        self.volumes[m] = self._canon(nib.load(p))
            else:
                # decision: fall back to raw scans so a half-processed subject still draws
                d = stripped_dir(shown)
                if not modality(d, shown, "T1").exists():
                    d, self.stripped = raw_dir(shown), False
                for m in MODALITIES:
                    p = modality(d, shown, m)
                    if p.exists():
                        self.volumes[m] = self._canon(nib.load(p))
        else:
            self.volumes["T1"] = self._canon(nib.load(out_dir(shown) / "t1_in_mni.nii.gz"))
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
        # decision: radiological view, patient left on the viewer's right, so flip RAS x
        x, y, z = self.cursor
        planes = []
        later = self.segs.get("2", np.zeros_like(self.seg))
        for volume in (self._gray(), self.seg, later):
            if which == "axial":
                planes.append(volume[::-1, ::-1, z].T)
            elif which == "coronal":
                planes.append(volume[::-1, y, ::-1].T)
            else:
                planes.append(volume[x, ::-1, ::-1].T)
        return planes

    def _marker(self, which):
        x, y, z = self.cursor
        nx, ny, nz = self.shape
        if which == "axial":
            return nx - 1 - x, ny - 1 - y
        if which == "coronal":
            return nx - 1 - x, nz - 1 - z
        return ny - 1 - y, nz - 1 - z

    def canon_mask(self, mask):
        name = "seg_in_mni.nii.gz" if self.space == "mni" else "seg.nii.gz"
        reference = nib.load(out_dir(self._grid()) / name)
        return self._canon(nib.Nifti1Image(mask.astype(np.uint8), reference.affine))[1] > 0

    def _target(self, kind, phase):
        if (kind, phase) not in self.targets:
            source = target_file(self._grid(), kind, phase, self.space)
            self.targets[kind, phase] = (self._canon(nib.load(source))[1] > 0) if source else None
        return self.targets[kind, phase]

    def plane(self, which, size):
        gray, lab, lab2 = self._slab(which)
        top = np.percentile(gray[gray > 0], 99) if (gray > 0).any() else 1
        rgb = np.repeat((np.clip(gray / max(top, 1) * 255, 0, 255)
                         ).astype(np.uint8)[:, :, None], 3, axis=2)

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

        # decision: targets as outlines, a fill hides what they grew from
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

        # decision: follow-up as outlines so both scans read at once
        if self.whole["2"] is not None:
            ring = outline(lab2 > 0)
            rgb[ring] = np.array(self.whole["2"], dtype=np.uint8)
        for value, color in TISSUE.items():
            if value == 0 or value in self.hidden["2"]:
                continue
            ring = outline(lab2 == value)
            if ring.any():
                rgb[ring] = np.array(color, dtype=np.uint8)

        mx, my = self._marker(which)
        if 0 <= my < rgb.shape[0]:
            rgb[my] = (rgb[my] * 0.45 + np.array(CROSS) * 0.55).astype(np.uint8)
        if 0 <= mx < rgb.shape[1]:
            rgb[:, mx] = (rgb[:, mx] * 0.45 + np.array(CROSS) * 0.55).astype(np.uint8)

        im = Image.fromarray(rgb)
        w, h = max(size[0], 8), max(size[1], 8)
        scale = min(w / im.width, h / im.height)
        out = fit(im, (w, h))
        # explainer: click mapping needs the letterbox offsets
        self.geom[which] = (scale, im.width, im.height,
                            (w - int(im.width * scale)) // 2, (h - int(im.height * scale)) // 2,
                            w, h)
        return out

    def click(self, which, fx, fy):
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
        # decision: coords only in MNI, SRI24's affine is a corner offset not a real origin
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
               "signal": {m: int(v[1][x, y, z]) for m, v in self.volumes.items()}}
        if self.space == "mni":
            world = nib.affines.apply_affine(self.affine, [x, y, z])
            out["coords"] = tuple(round(float(v)) for v in world)
            out["coord_space"] = "MNI mm"
        else:
            out["coords"] = None
            out["coord_space"] = "follow-up grid" if self.later else "SRI24 grid"
        return out
