#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Terminal UI for the segmentation pipeline. Needs kitty for the graphics protocol.

import asyncio
import json
import os
import re
import sys

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Grid, Vertical
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
ATLAS_FILES = [("YBA_696.nii", "696"), ("parcel_dict.csv", "ok"),
               ("parcelsynth.csv", "334"), ("top5k_tracts.obj", "3001")]


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

    def on_resize(self, event):
        self.app.schedule_draw()


class Pane(ImageWidget, Renderable=Renderable):
    # One slice plane: click or drag to move the crosshair.

    def __init__(self, which, **kwargs):
        super().__init__(**kwargs)
        self.which = which
        self.border_title = which

    def _seek(self, event):
        size = self.content_size
        if size.width and size.height:
            self.app.seek(self.which, event.x / size.width, event.y / size.height)

    def on_mouse_down(self, event):
        self.capture_mouse()
        self._seek(event)

    def on_mouse_up(self, event):
        self.release_mouse()

    def on_mouse_move(self, event):
        if event.button:
            self._seek(event)

    def on_resize(self, event):
        self.app.schedule_draw()


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
    .box { border: round #31312f; border-title-color: #bfa87f;
           border-title-style: bold; border-subtitle-color: #5b5a56; }
    #inputs { width: 32; }
    #mid { width: 1fr; }
    #right { width: 46; }
    #viewer { height: 1fr; }
    #log { height: 10; background: #141414; }
    #grid { layout: grid; grid-size: 2 2; grid-gutter: 0; height: 1fr; }
    #canvas { height: 1fr; }
    .pane { height: 1fr; border: round #262624; border-title-color: #8d8b85; }
    #readout { height: 1fr; border: round #262624; border-title-color: #8d8b85; padding: 0 1; }
    .title { color: #bfa87f; text-style: bold; }
    .dim { color: #8d8b85; }
    #subjects { height: auto; padding: 0 1; }
    #layers { height: auto; padding: 0 1; }
    #parcels { height: 1fr; background: #141414; }
    ListItem { background: #141414; padding: 0 1; }
    ListItem.--highlight { background: #262420; }
    #detail { height: 13; border-top: solid #31312f; padding: 0 1; }
    #listhead { height: 1; padding: 0 1; }
    """
    BINDINGS = [
        ("q", "quit", "quit"), ("v", "view", "3d/slices"), ("s", "space", "native/mni"),
        ("p", "palette", "palette"), ("f", "listing", "parcels/functions"),
        ("o", "overlay", "overlay"), ("h", "home", "reset"),
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
        self.scene = self.slices = self.proc = self._timer = None
        self.hidden_layers, self.hidden_parcels, self.subject_rows = set(), set(), {}

    # ---------------------------------------------------------------- compose

    def compose(self) -> ComposeResult:
        with Vertical(id="inputs", classes="box"):
            yield Rows(self.click_subject, id="subjects")
        with Vertical(id="mid"):
            with Vertical(id="viewer", classes="box"):
                yield Canvas(id="canvas")
                with Grid(id="grid"):
                    yield Pane("axial", id="p-axial", classes="pane")
                    yield Pane("coronal", id="p-coronal", classes="pane")
                    yield Pane("sagittal", id="p-sagittal", classes="pane")
                    yield Static("", id="readout")
            yield RichLog(id="log", wrap=True, markup=True, highlight=False, classes="box")
        with Vertical(id="right", classes="box"):
            yield Rows(self.click_layer, id="layers")
            yield Static("", id="listhead", classes="title")
            yield ListView(id="parcels")
            yield Static("", id="detail")
        yield Footer()

    def on_mount(self):
        self.query_one("#grid").display = False
        self.query_one("#inputs").border_title = "inputs"
        self.query_one("#inputs").border_subtitle = "click a subject"
        self.query_one("#log").border_title = "pipeline"
        self.query_one("#log").border_subtitle = "r run · c cancel"
        self.query_one("#right").border_title = "breakdown"
        self.query_one("#right").border_subtitle = "f parcels/functions · p palette"
        self.query_one("#readout").border_title = "cursor"
        self.log_write(f"[{ACCENT}]{len(self.results)} subjects[/] from {RESULTS}")
        if not os.environ.get("TERM", "").startswith("xterm-kitty"):
            self.log_write("[#f2c14e]not kitty: images will render as text blocks[/]")
        if self.subject:
            self.load_subject(self.subject)
            self.call_after_refresh(self.draw)

    def log_write(self, text):
        self.query_one("#log", RichLog).write(text)

    def schedule_draw(self):
        # resizing fires a burst of events; only redraw once it settles
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_timer(0.06, self.draw)

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
        out, row = [], 0
        self.subject_rows = {}
        out.append(f"[{ACCENT}]subjects[/]"); row += 1
        for s in self.subjects:
            d = self.results[s]
            a = d["atlas"]
            here = s == self.subject
            mark = "›" if here else " "
            style = ACCENT if here else DIM
            self.subject_rows[row] = s
            out.append(f"[{style}]{mark} {s.replace('UPENN-GBM-', ''):<11}{a['tumour_ml']:>7.0f} mL[/]")
            row += 1
            note = (f"dice {d['dice']['WT']}" if d.get("dice") else "no reference")
            self.subject_rows[row] = s
            out.append(f"[{FAINT}]    {len(a['parcels'])} parcels · {note}[/]")
            row += 1

        d = self.results[self.subject]
        r = d["regions_ml"]
        out += ["", f"[{ACCENT}]tumour[/]",
                f"[{DIM}] WT {r['WT']:>8} mL[/]", f"[{DIM}] TC {r['TC']:>8} mL[/]",
                f"[{DIM}] ET {r['ET']:>8} mL[/]",
                f"[{FAINT}] {d['atlas']['in_parcels_ml']} mL in parcels[/]"]
        out += ["", f"[{ACCENT}]scans[/]"]
        for m in MODALITIES:
            ok = (stripped_dir(self.subject) / f"{self.subject}_{m}.nii.gz").exists()
            out.append(f"[{DIM}] {m:<7}[/][{ACCENT if ok else FAINT}]"
                       f"{'stripped' if ok else 'missing':>12}[/]")
        out += ["", f"[{ACCENT}]atlas[/]"]
        for name, note in ATLAS_FILES:
            out.append(f"[{DIM}] {name:<18}[/][{FAINT}]{note:>6}[/]")
        self.query_one("#subjects", Rows).update("\n".join(out))

    def click_subject(self, row):
        chosen = self.subject_rows.get(row)
        if chosen and chosen != self.subject:
            self.load_subject(chosen)

    # ----------------------------------------------------------------- layers

    def refresh_layers(self):
        out = []
        volumes = self.results[self.subject]["labels_ml"]
        for key, name, colour in LAYERS:
            on = key not in self.hidden_layers
            value = volumes.get(key.split(":")[-1])
            value = f"{value} mL" if value else ""
            style = colour if on else FAINT
            out.append(f"[{style}]{'[×]' if on else '[ ]'}[/] "
                       f"[{DIM if on else FAINT}]{name:<17}{value:>9}[/]")
        pct = int(self.shell_opacity * 100)
        out.append(f"[{FAINT}]shell[/] [{DIM}]{bar(self.shell_opacity)}[/] [{FAINT}]{pct:>3}%[/]")
        self.query_one("#layers", Rows).update("\n".join(out))

    def click_layer(self, row):
        if 0 <= row < len(LAYERS):
            key = LAYERS[row][0]
            self.hidden_layers ^= {key}
            self.scene.set_visible(key, key not in self.hidden_layers)
            self.refresh_layers()
            if self.mode == "3d":
                self.draw()

    # ------------------------------------------------------------------- list

    def refresh_list(self):
        d = self.results[self.subject]
        listing = self.query_one("#parcels", ListView)
        listing.clear()
        if self.listing == "parcels":
            rows = d["atlas"]["parcels"]
            self.query_one("#listhead", Static).update(f"parcels involved · {len(rows)}")
            for p in rows:
                on = p["parcel"] not in self.hidden_parcels
                style = "" if on else f"[{FAINT}]"
                listing.append(ListItem(Label(
                    f"{style}{'[×]' if on else '[ ]'} {p['parcel']:<11}"
                    f"{bar(p['fraction'])} {p['fraction']:>4.0%}")))
        else:
            top = d.get("function", [])
            self.query_one("#listhead", Static).update(f"functions at risk · {len(top)}")
            peak = max((f["score"] for f in top), default=1)
            for f in top:
                listing.append(ListItem(Label(
                    f" {f['term']:<16}{bar(f['score'] / peak)} {f['score']:>5.2f}")))
        self.refresh_detail(0)

    def refresh_detail(self, index):
        d = self.results[self.subject]
        if self.listing == "parcels":
            rows = d["atlas"]["parcels"]
            if not rows or index >= len(rows):
                return
            p = rows[index]
            lines = [f"[{ACCENT}]{p['long_name'][:42]}[/]",
                     f"[{FAINT}]{p['lobe']} · {p['gyrus'][:26]}[/]",
                     f"[{DIM}]{p['in_tumour_ml']} of {p['parcel_ml']} mL · {p['tissue']}[/]", ""]
            terms = function.for_parcel(p["parcel"], 6)
            peak = max((v for _, v in terms), default=1)
            lines.append(f"[{FAINT}]parcelsynth[/]")
            lines += [f"[{DIM}]{t:<14}{bar(v / peak, 8)} {v:>5.2f}[/]" for t, v in terms]
        else:
            top = d.get("function", [])
            if not top or index >= len(top):
                return
            f = top[index]
            lines = [f"[{ACCENT}]{f['term']}[/]",
                     f"[{DIM}]weighted score {f['score']}[/]", "",
                     f"[{FAINT}]summed over the involved parcels,[/]",
                     f"[{FAINT}]each weighted by how much of it[/]",
                     f"[{FAINT}]the tumour occupies[/]", "",
                     f"[{FAINT}]meta-analytic association from[/]",
                     f"[{FAINT}]published studies, not this[/]",
                     f"[{FAINT}]patient's anatomy[/]"]
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
        self.refresh_list()
        self.query_one("#parcels", ListView).index = index
        self.draw()

    # ----------------------------------------------------------------- render

    def draw(self):
        if not self.scene:
            return
        cell = get_cell_size()
        viewer = self.query_one("#viewer")
        if self.mode == "3d":
            widget = self.query_one("#canvas", Canvas)
            size = widget.content_size
            w, h = max(size.width * cell.width, 64), max(size.height * cell.height, 64)
            self.scene.resize(w, h)
            widget.image = self.scene.image()
            viewer.border_title = f"viewer · 3d · {self.palette}"
            viewer.border_subtitle = "drag rotate · scroll zoom · v slices · h reset"
        else:
            for which in render.Slices.PLANES:
                pane = self.query_one(f"#p-{which}", Pane)
                size = pane.content_size
                pane.image = self.slices.plane(
                    which, (max(size.width * cell.width, 32), max(size.height * cell.height, 32)))
            self.refresh_readout()
            r = self.slices.readout()
            viewer.border_title = f"viewer · slices · {r['space']} · {r['modality']}"
            viewer.border_subtitle = "click to move · 1-4 modality · s space · o overlay · v 3d"

    def refresh_readout(self):
        r = self.slices.readout()
        lines = [f"[{FAINT}]{r['coord_space']}[/]",
                 f"[{DIM}]{r['coords'] if r['coords'] else '—'}[/]", "",
                 f"[{FAINT}]voxel[/]", f"[{DIM}]{r['voxel']}[/]", "",
                 f"[{FAINT}]label[/]", f"[{ACCENT}]{r['label']}[/]", "",
                 f"[{FAINT}]intensity[/]", f"[{DIM}]{r['intensity']}[/]", "",
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

    def _shell(self, delta):
        self.shell_opacity = max(0.0, min(1.0, self.shell_opacity + delta))
        self.scene.set_opacity("shell", self.shell_opacity)
        self.refresh_layers()
        self.draw()

    def action_dimmer(self):
        self._shell(-0.05)

    def action_brighter(self):
        self._shell(0.05)

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
