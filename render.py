# Offscreen rendering for the terminal UI: 3D scenes and slice planes, PIL images out.

import nibabel as nib
import numpy as np
import pyvista as pv
import trimesh
from PIL import Image

import atlas
import view
from paths import LABELS, MODALITIES, modality, out_dir, stripped_dir

pv.OFF_SCREEN = True

BG = "#141414"
SHELL = "#8f8d87"
TISSUE = {1: (242, 193, 78), 2: (63, 167, 214), 3: (228, 87, 46), 4: (142, 125, 190)}
_hex = lambda t: "#%02x%02x%02x" % tuple(t)


class Scene:
    # A persistent plotter; only the camera moves between frames.

    def __init__(self, subject, overlap, palette="contrast", top=30, size=(700, 700)):
        self.plotter = pv.Plotter(off_screen=True, window_size=list(size))
        self.plotter.set_background(BG)
        self.actors = {}

        info, surf = atlas.parcels(), atlas.meshes()
        index = {p["name"]: i for i, p in enumerate(info)}
        colours = atlas.colours(palette)
        involved = [r["parcel"] for r in overlap["parcels"]]

        rest = [surf[i] for n, i in index.items() if n not in set(involved)]
        if rest:
            shell = trimesh.util.concatenate(rest)
            self.actors["shell"] = self._add(shell.vertices, shell.faces, SHELL, 0.15)

        for name in involved[:top]:
            i = index[name]
            self.actors["P:" + name] = self._add(
                surf[i].vertices, surf[i].faces, _hex(colours[i]), 0.95)

        image = nib.load(out_dir(subject) / "seg_in_mni.nii.gz")
        data = np.asarray(image.dataobj).astype(np.int16)
        for value, name in LABELS.items():
            s = view._tumour_surface(data, image.affine, value)
            if s:
                self.actors["T:" + name] = self._add(s[0], s[1], _hex(TISSUE[value]), 1.0)

        self.plotter.show(auto_close=False)

    def _add(self, v, f, colour, opacity):
        f = np.asarray(f)
        poly = pv.PolyData(np.asarray(v), np.hstack([np.full((len(f), 1), 3), f]).ravel())
        return self.plotter.add_mesh(poly, color=colour, opacity=opacity, smooth_shading=True)

    def resize(self, width, height):
        if [width, height] != list(self.plotter.window_size):
            self.plotter.window_size = [max(width, 32), max(height, 32)]

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

    def image(self):
        self.plotter.render()
        return Image.fromarray(self.plotter.screenshot(return_img=True))


class Slices:
    # Orthogonal planes. Volumes are reoriented to RAS so plane logic is uniform.

    PLANES = ("axial", "coronal", "sagittal")

    def __init__(self, subject, space="native"):
        self.subject, self.space = subject, space
        self.volumes, self.seg = {}, None
        self.load(space)
        self.modality = "T1GD" if "T1GD" in self.volumes else next(iter(self.volumes))
        self.cursor = [s // 2 for s in self.shape]

    def load(self, space):
        self.space = space
        self.volumes = {}
        if space == "native":
            d = stripped_dir(self.subject)
            for m in MODALITIES:
                p = modality(d, self.subject, m)
                if p.exists():
                    self.volumes[m] = self._canon(nib.load(p))
            seg = out_dir(self.subject) / "seg.nii.gz"
        else:
            p = out_dir(self.subject) / "t1_in_mni.nii.gz"
            self.volumes["T1"] = self._canon(nib.load(p))
            seg = out_dir(self.subject) / "seg_in_mni.nii.gz"
        self.affine, first = next(iter(self.volumes.values()))
        self.shape = first.shape
        self.seg = self._canon(nib.load(seg))[1]
        self.modality = "T1GD" if "T1GD" in self.volumes else next(iter(self.volumes))
        self.cursor = [s // 2 for s in self.shape]

    @staticmethod
    def _canon(image):
        canonical = nib.as_closest_canonical(image)
        return canonical.affine, np.asarray(canonical.dataobj)

    def _grey(self):
        return self.volumes[self.modality][1]

    def plane(self, which, size):
        # radiological: patient left on the viewer's right. RAS x increases right, so flip it.
        x, y, z = self.cursor
        g, s = self._grey(), self.seg
        if which == "axial":
            grey, lab = g[::-1, ::-1, z].T, s[::-1, ::-1, z].T
        elif which == "coronal":
            grey, lab = g[::-1, y, ::-1].T, s[::-1, y, ::-1].T
        else:
            grey, lab = g[x, ::-1, ::-1].T, s[x, ::-1, ::-1].T

        top = np.percentile(grey[grey > 0], 99) if (grey > 0).any() else 1
        rgb = np.repeat((np.clip(grey / max(top, 1) * 255, 0, 255)
                         ).astype(np.uint8)[:, :, None], 3, axis=2)
        for value, colour in TISSUE.items():
            m = lab == value
            if m.any():
                rgb[m] = (rgb[m] * 0.55 + np.array(colour) * 0.45).astype(np.uint8)

        im = Image.fromarray(rgb)
        w, h = size
        scale = min(w / im.width, h / im.height)
        return im.resize((max(int(im.width * scale), 1), max(int(im.height * scale), 1)),
                         Image.NEAREST)

    def marks(self, which):
        return {"axial": ("R", "L", "A", "P"), "coronal": ("R", "L", "S", "I"),
                "sagittal": ("A", "P", "S", "I")}[which]

    def readout(self):
        # only MNI has an anatomical origin; the SRI24 affine is a corner offset,
        # so reporting it as coordinates would be misleading
        x, y, z = self.cursor
        label = int(self.seg[x, y, z])
        out = {"voxel": (x, y, z), "label": LABELS.get(label, "background"),
               "intensity": int(self._grey()[x, y, z]),
               "space": self.space, "modality": self.modality}
        if self.space == "mni":
            world = nib.affines.apply_affine(self.affine, [x, y, z])
            out["coords"] = tuple(round(float(v)) for v in world)
            out["coord_space"] = "MNI mm"
        else:
            out["coords"] = None
            out["coord_space"] = "SRI24 grid, no anatomical origin"
        return out
