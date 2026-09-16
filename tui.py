#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Terminal UI for the segmentation pipeline. Needs kitty for the graphics protocol.

import asyncio
import json
import os
import re
import subprocess
import sys

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import Footer, Input, Label, ListItem, ListView, RichLog, Static
from textual_image._terminal import get_cell_size
from textual_image.renderable import Image as Renderable
from textual_image.widget import Image as ImageWidget

import function
import render
from paths import ATLAS_INPUT, MODALITIES, RESULTS, ROOT, stripped_dir, subject_id

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")

# gruvbox dark, the same values btop reads out of gruvbox_dark_v2.theme
FG, DIM, FAINT = "#ebdbb2", "#a89984", "#7c6f64"
OK, WARN, BAD = "#b8bb26", "#fabd2f", "#fb4934"
SHELL_OPACITY = 0.15

LAYERS = [("T:NETC", "NETC necrosis", "#f2c14e"), ("T:SNFH", "SNFH oedema", "#3fa7d6"),
          ("T:ET", "ET enhancing", "#e4572e"), ("tracts", "tracts crossing", "#689d6a"),
          ("shell", "atlas shell", "#a89984")]
ATLAS_FILES = [("YBA_696.nii", "volume", "696"),
               ("YBA_696.glb", "meshes", "glb"),
               ("top5k_tracts_decimated.obj", "tracts", "3001"),
               ("parcelsynth_activations_functionalterms.csv", "parcelsynth", "334"),
               ("structuralconnectivity_metrics_mean.csv", "connectivity", "696²")]
IDLE = "type a command · esc leaves · t returns"
HELP = ["main.py <n>...   segment and map a subject",
        "view.py <n>      open the browser view",
        "cancel           stop the running job",
        "clear            clear this log"]


def bar(fraction, n=10):
    filled = round(max(0.0, min(1.0, fraction)) * n)
    return "█" * filled + "░" * (n - filled)


def row(left, right, width=30):
    # One list line with the right column flush to the edge.
    room = width - len(right)
    return f"{left[:room]:<{room}}{right}"


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

    def __init__(self, handler=None, **kwargs):
        super().__init__("", **kwargs)
        self.handler = handler

    def on_click(self, event):
        if self.handler:
            self.handler(event.y)


class Prompt(Input):
    # Escape gives the keyboard back, up and down walk the history.
    BINDINGS = [("escape", "leave", "leave the shell")]

    def action_leave(self):
        self.app.leave_shell()

    def on_key(self, event):
        if event.key == "escape":
            self.app.leave_shell()
        elif event.key == "up":
            self.app.recall(-1)
        elif event.key == "down":
            self.app.recall(1)
        else:
            return
        event.stop()
        event.prevent_default()


class Console(App):
    CSS = """
    Screen { background: #282828; }
    #top { height: 1fr; }
    .box { border: round #a89984; border-title-color: #ebdbb2;
           border-title-style: bold; border-subtitle-color: #7c6f64;
           background: #282828; padding: 0; }
    .card { border: round #665c54; border-title-color: #a89984;
            height: auto; padding: 0 1; background: #282828; }
    #inputs { width: 34; padding: 0; overflow-y: auto; overflow-x: hidden;
              scrollbar-size-vertical: 1; }
    #viewer { width: 1fr; }
    #right { width: 46; }
    #grid { layout: grid; grid-size: 2 2; grid-gutter: 0; height: 1fr; }
    #canvas { height: 1fr; }
    .pane { height: 1fr; border: round #504945; border-title-color: #a89984; }
    #readout { height: 1fr; border: round #504945; border-title-color: #a89984;
               padding: 0 1; }
    #shell { height: 14; }
    #shell:focus-within { border: round #fabd2f; }
    #log { height: 1fr; background: #282828; scrollbar-size-vertical: 1;
           padding: 0 1; }
    #line { height: 1; padding: 0 1; }
    #sigil { width: 2; color: #b8bb26; text-style: bold; }
    #cmd { border: none; height: 1; padding: 0; background: #282828; color: #ebdbb2; }
    #cmd:focus { border: none; background: #32302f; }
    #layers { height: auto; padding: 0 1; }
    #parcels { height: 1fr; background: #282828; scrollbar-size-vertical: 1; }
    ListItem { background: #282828; padding: 0 1; }
    ListItem.--highlight { background: #32302f; color: #d3869b; }
    #detail { height: 13; border-top: solid #504945; padding: 0 1; }
    #listhead { height: 1; padding: 0 1; color: #ebdbb2; text-style: bold;
                border-top: solid #504945; }
    #inputs, #log, #parcels {
        scrollbar-background: #282828; scrollbar-color: #504945;
        scrollbar-background-hover: #282828; scrollbar-color-hover: #7c6f64;
        scrollbar-background-active: #282828; scrollbar-color-active: #a89984; }
    Footer { background: #282828; }
    FooterKey { background: #282828; }
    FooterKey .footer-key--key { color: #cc241d; background: #282828; }
    FooterKey .footer-key--description { color: #a89984; background: #282828; }
    """
    BINDINGS = [
        ("q", "quit", "quit"), ("v", "view", "3d/slices"), ("s", "space", "native/mni"),
        ("p", "palette", "palette"), ("f", "listing", "parcels/functions"),
        ("o", "overlay", "overlay"), ("h", "home", "reset"),
        ("t", "shell", "terminal"), ("c", "cancel", "cancel"),
    ]

    def __init__(self):
        super().__init__()
        self.results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        self.subjects = sorted(self.results)
        self.subject = self.subjects[0] if self.subjects else None
        self.mode, self.space, self.palette, self.listing = "3d", "native", "contrast", "parcels"
        self.scene = self.slices = self.proc = self._timer = None
        self.hidden_layers, self.hidden_parcels, self.subject_rows = set(), set(), {}
        self.history, self.hpos = [], 0
        self._panel_width = self._list_width = 0

    # ---------------------------------------------------------------- compose

    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            with Vertical(id="inputs"):
                yield Rows(self.click_subject, id="subjects", classes="card")
                yield Rows(id="tumour", classes="card")
                yield Rows(id="scans", classes="card")
                yield Rows(id="atlasfiles", classes="card")
            with Vertical(id="viewer", classes="box"):
                yield Canvas(id="canvas")
                with Grid(id="grid"):
                    yield Pane("axial", id="p-axial", classes="pane")
                    yield Pane("coronal", id="p-coronal", classes="pane")
                    yield Pane("sagittal", id="p-sagittal", classes="pane")
                    yield Static("", id="readout")
            with Vertical(id="right", classes="box"):
                yield Rows(self.click_layer, id="layers")
                yield Static("", id="listhead")
                yield ListView(id="parcels")
                yield Static("", id="detail")
        with Vertical(id="shell", classes="box"):
            yield RichLog(id="log", wrap=True, markup=True, highlight=False)
            with Horizontal(id="line"):
                yield Static("$", id="sigil")
                yield Prompt(id="cmd", placeholder="./main.py 36")
        yield Footer()

    def on_mount(self):
        self.query_one("#grid").display = False
        for widget, title in [("#subjects", "subjects"), ("#tumour", "tumour"),
                              ("#scans", "scans"), ("#atlasfiles", "atlas")]:
            self.query_one(widget).border_title = title
        self.query_one("#shell").border_title = "pipeline"
        self.query_one("#shell").border_subtitle = IDLE
        self.query_one("#right").border_title = "breakdown"
        self.query_one("#right").border_subtitle = "f parcels/functions · p palette"
        self.query_one("#readout").border_title = "cursor"
        self.log_write(f"[{DIM}]{len(self.results)} subjects in {RESULTS.name}. "
                       f"type help for commands.[/]")
        if not os.environ.get("TERM", "").startswith("xterm-kitty"):
            self.log_write(f"[{WARN}]not kitty: images will render as text blocks[/]")
        if self.subject:
            self.load_subject(self.subject)
            self.call_after_refresh(self.draw)
        self.set_focus(self.query_one("#parcels"))

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
        self.scene = render.Scene(subject, overlap, palette=self.palette)
        self.scene.set_opacity("shell", SHELL_OPACITY)
        self.slices = render.Slices(subject, self.space)
        self.refresh_inputs()
        self.refresh_layers()
        self.refresh_list()
        self.draw()

    def panel_width(self):
        # The cards lose a column whenever the side scrollbar appears.
        width = self.query_one("#subjects", Rows).size.width
        return width if width > 12 else 28

    def refresh_inputs(self):
        w = self._panel_width = self.panel_width()
        out = []
        self.subject_rows = {}
        for n, s in enumerate(self.subjects):
            a = self.results[s]["atlas"]
            here = s == self.subject
            self.subject_rows[n] = s
            left = ("› " if here else "  ") + s[10:]
            right = "%3dp %7.0f mL" % (len(a["parcels"]), a["tumour_ml"])
            out.append(f"[{FG if here else DIM}]{row(left, right, w)}[/]")
        self.query_one("#subjects", Rows).update("\n".join(out))

        d = self.results[self.subject]
        r = d["regions_ml"]
        lines = [f"[{DIM}]{row(name, f'{r[name]} mL', w)}[/]" for name in ("WT", "TC", "ET")]
        in_parcels = f"{d['atlas']['in_parcels_ml']} mL"
        lines.append(f"[{FAINT}]{row('in parcels', in_parcels, w)}[/]")
        dice = f"{d['dice']['WT']:.2f}" if d.get("dice") else "no reference"
        lines.append(f"[{FAINT}]{row('dice WT', dice, w)}[/]")
        self.query_one("#tumour", Rows).update("\n".join(lines))

        lines = []
        for m in MODALITIES:
            ok = (stripped_dir(self.subject) / f"{self.subject}_{m}.nii.gz").exists()
            state = "stripped" if ok else "missing"
            lines.append(f"[{DIM}]{m}[/][{OK if ok else BAD}]"
                         f"{state:>{w - len(m)}}[/]")
        self.query_one("#scans", Rows).update("\n".join(lines))

        lines = []
        for name, label, note in ATLAS_FILES:
            have = (ATLAS_INPUT / name).exists()
            state = note if have else "missing"
            lines.append(f"[{DIM if have else FAINT}]{label}[/]"
                         f"[{FAINT if have else BAD}]{state:>{w - len(label)}}[/]")
        self.query_one("#atlasfiles", Rows).update("\n".join(lines))

    def click_subject(self, row_index):
        chosen = self.subject_rows.get(row_index)
        if chosen and chosen != self.subject:
            self.load_subject(chosen)

    # ----------------------------------------------------------------- layers

    def refresh_layers(self):
        out = []
        w = max(self.query_one("#layers", Rows).size.width, 30)
        volumes = self.results[self.subject]["labels_ml"]
        for key, name, colour in LAYERS:
            on = key not in self.hidden_layers
            value = volumes.get(key.split(":")[-1])
            value = f"{value} mL" if value else ""
            out.append(f"[{colour if on else FAINT}]{'[×]' if on else '[ ]'}[/] "
                       f"[{DIM if on else FAINT}]{row(name, value, w - 4)}[/]")
        self.query_one("#layers", Rows).update("\n".join(out))

    def click_layer(self, row_index):
        if 0 <= row_index < len(LAYERS):
            key = LAYERS[row_index][0]
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
        w = max(listing.size.width - 3, 34)
        if self.listing == "parcels":
            rows = d["atlas"]["parcels"]
            self.query_one("#listhead", Static).update(f"parcels involved · {len(rows)}")
            for p in rows:
                on = p["parcel"] not in self.hidden_parcels
                style = "" if on else f"[{FAINT}]"
                left = f"{'[×]' if on else '[ ]'} {p['parcel']}"
                right = f"{bar(p['fraction'])} {p['fraction']:>4.0%}"
                listing.append(ListItem(Label(style + row(left, right, w))))
        else:
            top = d.get("function", [])
            self.query_one("#listhead", Static).update(f"functions at risk · {len(top)}")
            peak = max((f["score"] for f in top), default=1)
            for f in top:
                right = f"{bar(f['score'] / peak)} {f['score']:>5.2f}"
                listing.append(ListItem(Label(row(" " + f["term"], right, w))))
        self.refresh_detail(0)

    def refresh_detail(self, index):
        d = self.results[self.subject]
        if self.listing == "parcels":
            rows = d["atlas"]["parcels"]
            if not rows or index >= len(rows):
                return
            p = rows[index]
            lines = [f"[{FG}]{p['long_name'][:42]}[/]",
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
            lines = [f"[{FG}]{f['term']}[/]",
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
        if self.panel_width() != self._panel_width:
            self.refresh_inputs()
            self.refresh_layers()
        listing = self.query_one("#parcels", ListView)
        if listing.size.width != self._list_width:
            self._list_width = listing.size.width
            index = listing.index
            self.refresh_list()
            listing.index = index
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
                 f"[{FAINT}]label[/]", f"[{FG}]{r['label']}[/]", "",
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

    def action_shell(self):
        self.set_focus(self.query_one("#cmd"))

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

    # ------------------------------------------------------------------ shell

    def leave_shell(self):
        self.set_focus(self.query_one("#parcels"))

    def recall(self, delta):
        if not self.history:
            return
        self.hpos = max(0, min(len(self.history), self.hpos + delta))
        field = self.query_one("#cmd", Prompt)
        field.value = self.history[self.hpos] if self.hpos < len(self.history) else ""
        field.cursor_position = len(field.value)

    @on(Input.Submitted, "#cmd")
    def submit(self, event):
        line = event.value.strip()
        field = self.query_one("#cmd", Prompt)
        field.value = ""
        self.log_write(f"[{OK}]$[/] [{FG}]{line}[/]")
        if not line:
            return
        self.history.append(line)
        self.hpos = len(self.history)
        self.execute(line)

    def execute(self, line):
        parts = line.split()
        name = parts[0].lstrip("./")
        numbers = []
        for arg in parts[1:]:
            if not arg.isdigit():
                self.log_write(f"[{BAD}]{arg}: expected a subject number[/]")
                return
            numbers.append(int(arg))

        if name in ("main.py", "main"):
            if not numbers:
                self.log_write(f"[{BAD}]usage: ./main.py <subject numbers>[/]")
            elif self.proc:
                self.log_write(f"[{WARN}]a job is already running, cancel it first[/]")
            else:
                self.run_pipeline(numbers)
        elif name in ("view.py", "view"):
            self.open_view(numbers)
        elif name in ("cancel", "stop"):
            self.action_cancel()
        elif name == "clear":
            self.query_one("#log", RichLog).clear()
        elif name in ("quit", "exit"):
            self.exit()
        elif name == "help":
            for text in HELP:
                self.log_write(f"[{DIM}]{text}[/]")
        else:
            self.log_write(f"[{BAD}]{name}: not a command. try help.[/]")

    def open_view(self, numbers):
        wanted = subject_id(numbers[0]) if numbers else self.subject
        if wanted not in self.results:
            self.log_write(f"[{BAD}]{wanted} has no results yet[/]")
            return
        self.log_write(f"[{DIM}]opening {wanted} in the browser[/]")
        subprocess.Popen([sys.executable, "view.py", wanted.split("-")[2].split("_")[0]],
                         cwd=str(ROOT), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)

    def action_cancel(self):
        if self.proc:
            self.proc.terminate()
            self.log_write(f"[{WARN}]cancelled[/]")
        else:
            self.log_write(f"[{FAINT}]nothing running[/]")

    def busy(self, note):
        self.query_one("#shell").border_subtitle = note

    @work(exclusive=True)
    async def run_pipeline(self, numbers):
        self.busy("running · c cancels")
        # rich draws a live spinner that becomes escape-code debris in a log widget
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "main.py", *[str(n) for n in numbers], cwd=str(ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        async for raw in self.proc.stdout:
            text = ANSI.sub("", raw.decode(errors="replace")).rstrip()
            if text:
                self.log_write(f"[{DIM}]{text}[/]")
        code = await self.proc.wait()
        self.proc = None
        self.busy(IDLE)
        if code < 0:
            self.log_write(f"[{WARN}]stopped[/]")
            return
        if code != 0:
            self.log_write(f"[{BAD}]exited {code}[/]")
            return
        self.results = json.loads(RESULTS.read_text())
        self.subjects = sorted(self.results)
        self.log_write(f"[{OK}]done[/]")
        self.load_subject(subject_id(numbers[-1]))


if __name__ == "__main__":
    Console().run()
