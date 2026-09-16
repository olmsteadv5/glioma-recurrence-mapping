#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Terminal UI for the segmentation pipeline. Needs kitty for the graphics protocol.

import asyncio
import json
import os
import re
import sys

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Grid, Vertical
from textual.widgets import Footer, Label, ListItem, ListView, RichLog, Static
from textual_image._terminal import get_cell_size
from textual_image.renderable import Image as Renderable
from textual_image.widget import Image as ImageWidget

import function
import render
from paths import MODALITIES, RESULTS, ROOT, stripped_dir, subject_id

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")

ACCENT, DIM, FAINT = "#bfa87f", "#8d8b85", "#5b5a56"
LAYERS = [("T:NETC", "NETC necrosis", "#f2c14e"), ("T:SNFH", "SNFH oedema", "#3fa7d6"),
          ("T:ET", "ET enhancing", "#e4572e"), ("tracts", "tracts crossing", "#5f9b72"),
          ("shell", "atlas shell", "#8f8d87")]


def bar(fraction, n=10):
    filled = round(max(0.0, min(1.0, fraction)) * n)
    return "█" * filled + "░" * (n - filled)


class Canvas(ImageWidget, Renderable=Renderable):
    # 3D view: drag to rotate, scroll to zoom.

    def on_mouse_down(self, event):
        self.capture_mouse()

    def on_mouse_up(self, event):
        self.release_mouse()

    def on_mouse_move(self, event):
        if event.button:
            self.app.rotate_scene(-event.delta_x * 3, event.delta_y * 3)

    def on_mouse_scroll_down(self, event):
        self.app.zoom_scene(0.9)

    def on_mouse_scroll_up(self, event):
        self.app.zoom_scene(1.1)


class Pane(ImageWidget, Renderable=Renderable):
    # One slice plane: click or drag to move the crosshair.

    def __init__(self, which, **kwargs):
        super().__init__(**kwargs)
        self.which = which

    def _seek(self, event):
        if self.size.width and self.size.height:
            self.app.seek(self.which, event.x / self.size.width, event.y / self.size.height)

    def on_mouse_down(self, event):
        self.capture_mouse()
        self._seek(event)

    def on_mouse_up(self, event):
        self.release_mouse()

    def on_mouse_move(self, event):
        if event.button:
            self._seek(event)


class Rows(Static):
    # Click-addressable list; the app decides what a row click means.

    def __init__(self, handler, **kwargs):
        super().__init__("", **kwargs)
        self.handler = handler

    def on_click(self, event):
        self.handler(event.y)


class Console(App):
    CSS = """
    Screen { layout: horizontal; background: #141414; }
    #inputs { width: 30; border: round #31312f; }
    #mid { width: 1fr; }
    #right { width: 46; border: round #31312f; }
    #viewer { height: 1fr; border: round #31312f; }
    #log { height: 11; border: round #31312f; background: #141414; }
    #grid { layout: grid; grid-size: 2 2; grid-gutter: 0; height: 1fr; }
    #canvas { height: 1fr; }
    .pane { height: 1fr; border: round #262624; }
    #readout { height: 1fr; border: round #262624; padding: 1; }
    #caption { height: 1; color: #8d8b85; }
    .title { color: #bfa87f; text-style: bold; }
    .dim { color: #8d8b85; }
    #subjects { height: 9; background: #141414; }
    ListItem { background: #141414; padding: 0 1; }
    ListItem.--highlight { background: #262420; }
    #parcels { height: 1fr; background: #141414; }
    #detail { height: 11; border-top: solid #31312f; padding: 0 1; }
    """
    BINDINGS = [
        ("q", "quit", "quit"), ("v", "view", "3d/slices"), ("s", "space", "native/mni"),
        ("p", "palette", "palette"), ("f", "listing", "parcels/functions"),
        ("o", "overlay", "overlay"), ("h", "home", "reset view"),
        ("bracketleft", "dimmer", "shell -"), ("bracketright", "brighter", "shell +"),
        ("r", "run", "run"), ("c", "cancel", "cancel"),
    ]

    def __init__(self):
        super().__init__()
        self.results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        self.subjects = sorted(self.results)
        self.subject = self.subjects[0] if self.subjects else None
        self.mode, self.space, self.palette, self.listing = "3d", "native", "contrast", "parcels"
        self.shell_opacity = 0.15
        self.scene = self.slices = self.proc = None
        self.hidden_layers, self.hidden_parcels = set(), set()

    # ---------------------------------------------------------------- compose

    def compose(self) -> ComposeResult:
        with Vertical(id="inputs"):
            yield Label(" subjects", classes="title")
            yield ListView(*[ListItem(Label(s.replace("UPENN-GBM-", "")))
                             for s in self.subjects], id="subjects")
            yield Static("", id="meta", classes="dim")
        with Vertical(id="mid"):
            with Vertical(id="viewer"):
                yield Canvas(id="canvas")
                with Grid(id="grid"):
                    yield Pane("axial", id="p-axial", classes="pane")
                    yield Pane("coronal", id="p-coronal", classes="pane")
                    yield Pane("sagittal", id="p-sagittal", classes="pane")
                    yield Static("", id="readout")
                yield Static("", id="caption")
            yield RichLog(id="log", wrap=True, markup=True, highlight=False)
        with Vertical(id="right"):
            yield Rows(self.click_layer, id="layers")
            yield Static("", id="listhead", classes="title")
            yield ListView(id="parcels")
            yield Static("", id="detail")
        yield Footer()

    def on_mount(self):
        self.query_one("#grid").display = False
        self.log_write(f"[{ACCENT}]{len(self.results)} subjects[/] from {RESULTS}")
        if not os.environ.get("TERM", "").startswith("xterm-kitty"):
            self.log_write(f"[#f2c14e]not kitty: images will render as text blocks[/]")
        if self.subject:
            self.load_subject(self.subject)
            # the viewer has no size until layout settles, so the first frame would be tiny
            self.call_after_refresh(self.draw)

    def log_write(self, text):
        self.query_one("#log", RichLog).write(text)

    # ---------------------------------------------------------------- subject

    def load_subject(self, subject):
        self.subject = subject
        self.hidden_layers, self.hidden_parcels = set(), set()
        overlap = self.results[subject]["atlas"]
        self.log_write(f"loading {subject}")
        self.scene = render.Scene(subject, overlap, palette=self.palette)
        self.scene.set_opacity("shell", self.shell_opacity)
        self.slices = render.Slices(subject, self.space)
        self.refresh_inputs()
        self.refresh_layers()
        self.refresh_list()
        self.draw()

    def refresh_inputs(self):
        d = self.results[self.subject]
        r, a = d["regions_ml"], d["atlas"]
        have = [m for m in MODALITIES
                if (stripped_dir(self.subject) / f"{self.subject}_{m}.nii.gz").exists()]
        lines = [f" WT {r['WT']:>7} mL", f" TC {r['TC']:>7} mL", f" ET {r['ET']:>7} mL", "",
                 f" {a['in_parcels_ml']} of {a['tumour_ml']} mL",
                 f" in {len(a['parcels'])} parcels"]
        if d.get("dice"):
            lines += ["", " dice " + "/".join(f"{v}" for v in d["dice"].values())]
        lines += ["", " scans"] + [f"   {m}  stripped" for m in have]
        self.query_one("#meta", Static).update("\n".join(lines))

    def refresh_layers(self):
        out = [f"[{ACCENT}] layers[/]"]
        volumes = self.results[self.subject]["labels_ml"]
        for key, name, colour in LAYERS:
            on = key not in self.hidden_layers
            mark = "[×]" if on else "[ ]"
            value = volumes.get(key.split(":")[-1], "")
            value = f"{value} mL" if value else ""
            style = colour if on else FAINT
            out.append(f" [{style}]{mark}[/] {name:<17}[{FAINT}]{value:>9}[/]")
        pct = int(self.shell_opacity * 100)
        out.append(f"\n[{ACCENT}] shell opacity[/]  [{DIM}]{bar(self.shell_opacity)} {pct:>3}%[/]")
        self.query_one("#layers", Rows).update("\n".join(out))

    def click_layer(self, row):
        if 1 <= row <= len(LAYERS):
            key = LAYERS[row - 1][0]
            self.hidden_layers ^= {key}
            self.scene.set_visible(key, key not in self.hidden_layers)
            self.refresh_layers()
            if self.mode == "3d":
                self.draw()

    # ------------------------------------------------------------------- list

    def refresh_list(self):
        d = self.results[self.subject]
        view_list = self.query_one("#parcels", ListView)
        view_list.clear()
        if self.listing == "parcels":
            self.query_one("#listhead", Static).update(
                f" parcels involved · {len(d['atlas']['parcels'])}")
            for p in d["atlas"]["parcels"]:
                on = p["parcel"] not in self.hidden_parcels
                mark = "[×]" if on else "[ ]"
                style = "" if on else f"[{FAINT}]"
                view_list.append(ListItem(Label(
                    f"{style}{mark} {p['parcel']:<11}{bar(p['fraction'])} {p['fraction']:>4.0%}")))
        else:
            self.query_one("#listhead", Static).update(" functions at risk")
            top = d.get("function", [])
            peak = max((f["enrichment"] for f in top), default=1)
            for f in top:
                view_list.append(ListItem(Label(
                    f" {f['term']:<17}{bar(f['enrichment'] / peak)} x{f['enrichment']:<5.1f}")))
        self.refresh_detail(0)

    def refresh_detail(self, index):
        d = self.results[self.subject]
        if self.listing == "parcels":
            rows = d["atlas"]["parcels"]
            if not rows or index >= len(rows):
                return
            p = rows[index]
            lines = [f"[{ACCENT}]{p['long_name']}[/]",
                     f"[{DIM}]{p['lobe']} · {p['gyrus']}[/]",
                     f"[{DIM}]{p['in_tumour_ml']} of {p['parcel_ml']} mL · {p['tissue']}[/]", ""]
            try:
                terms = function.for_parcel(p["parcel"], 6)
                lines.append(f"[{FAINT}]parcelsynth[/]")
                peak = max((v for _, v in terms), default=1)
                lines += [f"[{DIM}]{t:<14}{bar(v / peak, 8)} {v:>5.2f}[/]" for t, v in terms]
            except Exception as exc:
                lines.append(f"[{FAINT}]{exc}[/]")
        else:
            top = d.get("function", [])
            if not top or index >= len(top):
                return
            f = top[index]
            lines = [f"[{ACCENT}]{f['term']}[/]",
                     f"[{DIM}]x{f['enrichment']} over the whole-brain average[/]",
                     f"[{DIM}]weighted score {f['score']}[/]", "",
                     f"[{FAINT}]meta-analytic association from published[/]",
                     f"[{FAINT}]studies, not this patient's anatomy[/]"]
        self.query_one("#detail", Static).update("\n".join(lines))

    @on(ListView.Highlighted, "#parcels")
    def highlight(self, event):
        index = event.list_view.index or 0
        self.refresh_detail(index)
        if self.listing == "parcels" and self.mode == "3d" and self.scene:
            rows = self.results[self.subject]["atlas"]["parcels"]
            if index < len(rows):
                self.scene.isolate(rows[index]["parcel"])
                self.draw()

    @on(ListView.Selected, "#parcels")
    def select(self, event):
        if self.listing != "parcels":
            return
        index = event.list_view.index or 0
        rows = self.results[self.subject]["atlas"]["parcels"]
        if index >= len(rows):
            return
        name = rows[index]["parcel"]
        self.hidden_parcels ^= {name}
        self.scene.set_visible("P:" + name, name not in self.hidden_parcels)
        keep = index
        self.refresh_list()
        self.query_one("#parcels", ListView).index = keep
        self.draw()

    @on(ListView.Highlighted, "#subjects")
    def pick(self, event):
        if event.item is None:
            return
        chosen = self.subjects[event.list_view.index or 0]
        if chosen != self.subject:
            self.load_subject(chosen)

    # ----------------------------------------------------------------- render

    def draw(self):
        cell = get_cell_size()
        if self.mode == "3d":
            widget = self.query_one("#canvas", Canvas)
            w = max(widget.size.width * cell.width, 64)
            h = max(widget.size.height * cell.height, 64)
            self.scene.resize(w, h)
            widget.image = self.scene.image()
            self.query_one("#caption", Static).update(
                f" 3d · {self.palette} · drag rotates, scroll zooms")
        else:
            for which in render.Slices.PLANES:
                pane = self.query_one(f"#p-{which}", Pane)
                w = max(pane.size.width * cell.width, 32)
                h = max(pane.size.height * cell.height, 32)
                pane.image = self.slices.plane(which, (w, h))
            self.refresh_readout()
            r = self.slices.readout()
            self.query_one("#caption", Static).update(
                f" slices · {r['space']} · {r['modality']} · 1-4 modality, click to move")

    def refresh_readout(self):
        r = self.slices.readout()
        coords = f"{r['coords']}" if r["coords"] else "—"
        lines = [f"[{ACCENT}]cursor[/]", "",
                 f"[{FAINT}]{r['coord_space']}[/]", f"{coords}", "",
                 f"[{FAINT}]voxel[/]", f"{r['voxel']}", "",
                 f"[{FAINT}]label[/]", f"{r['label']}", "",
                 f"[{FAINT}]intensity[/]", f"{r['intensity']}", "",
                 f"[{FAINT}]{'  '.join(f'{i+1} {m}' for i, m in enumerate(r['modalities']))}[/]"]
        self.query_one("#readout", Static).update("\n".join(lines))

    def rotate_scene(self, azimuth, elevation):
        if self.mode == "3d" and self.scene:
            self.scene.rotate(azimuth, elevation)
            self.draw()

    def zoom_scene(self, factor):
        if self.mode == "3d" and self.scene:
            self.scene.zoom(factor)
            self.draw()

    def seek(self, which, fx, fy):
        if self.mode == "slices" and self.slices:
            self.slices.click(which, fx, fy)
            self.draw()

    def on_resize(self, event: events.Resize):
        if self.scene:
            self.call_after_refresh(self.draw)

    # ---------------------------------------------------------------- actions

    def action_view(self):
        self.mode = "slices" if self.mode == "3d" else "3d"
        self.query_one("#canvas").display = self.mode == "3d"
        self.query_one("#grid").display = self.mode == "slices"
        self.call_after_refresh(self.draw)

    def action_space(self):
        self.space = "mni" if self.space == "native" else "native"
        self.slices.load(self.space)
        self.log_write(f"slices now {self.space}")
        if self.mode == "slices":
            self.draw()

    def action_palette(self):
        self.palette = "yba" if self.palette == "contrast" else "contrast"
        self.scene.recolour(self.palette)
        self.draw()

    def action_listing(self):
        self.listing = "functions" if self.listing == "parcels" else "parcels"
        if self.scene:
            self.scene.unisolate()
        self.refresh_list()
        self.draw()

    def action_overlay(self):
        self.slices.overlay = not self.slices.overlay
        if self.mode == "slices":
            self.draw()

    def action_home(self):
        if self.scene:
            self.scene.reset()
            self.scene.unisolate()
            self.draw()

    def action_dimmer(self):
        self.shell_opacity = max(0.0, self.shell_opacity - 0.05)
        self.scene.set_opacity("shell", self.shell_opacity)
        self.refresh_layers()
        self.draw()

    def action_brighter(self):
        self.shell_opacity = min(1.0, self.shell_opacity + 0.05)
        self.scene.set_opacity("shell", self.shell_opacity)
        self.refresh_layers()
        self.draw()

    def on_key(self, event: events.Key):
        if self.mode == "slices" and self.slices:
            if event.key in "1234":
                names = self.slices.readout()["modalities"]
                i = int(event.key) - 1
                if i < len(names):
                    self.slices.modality = names[i]
                    self.draw()
            elif event.key in ("pageup", "pagedown"):
                self.slices.step(2, 1 if event.key == "pageup" else -1)
                self.draw()

    # --------------------------------------------------------------- pipeline

    def action_run(self):
        if self.proc:
            self.log_write("[#f2c14e]already running[/]")
            return
        self.run_pipeline(int(self.subject.split("-")[2].split("_")[0]))

    def action_cancel(self):
        if self.proc:
            self.proc.terminate()
            self.log_write("[#f2c14e]cancelled[/]")

    @work(exclusive=True)
    async def run_pipeline(self, number):
        self.log_write(f"[{ACCENT}]./main.py {number}[/]")
        # rich draws a live spinner that becomes escape-code debris in a log widget
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "main.py", str(number), cwd=str(ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        async for line in self.proc.stdout:
            text = ANSI.sub("", line.decode(errors="replace")).rstrip()
            if text:
                self.log_write(text)
        await self.proc.wait()
        self.proc = None
        self.results = json.loads(RESULTS.read_text())
        self.subjects = sorted(self.results)
        self.log_write("[#86a186]done[/]")
        self.load_subject(subject_id(number))


if __name__ == "__main__":
    Console().run()
