#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Terminal UI for the segmentation pipeline. Needs kitty for the graphics protocol.

import asyncio
import json
import os
import re
import subprocess
import sys
import time

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
OK, WARN, BAD, GONE = "#b8bb26", "#fabd2f", "#fb4934", "#504945"
SHELL_OPACITY = 0.15

LAYERS = [("T:NETC", "NETC necrosis", "#f2c14e"), ("T:SNFH", "SNFH oedema", "#3fa7d6"),
          ("T:ET", "ET enhancing", "#e4572e"), ("tracts", "tracts crossing", "#689d6a"),
          ("shell", "brain surface", "#a89984"),
          ("parcels", "involved parcels", "#d3869b"),
          ("focus", "dim the others", "#d3869b")]
ATLAS_FILES = [("YBA_696.nii", "volume", "696"),
               ("YBA_696.glb", "meshes", "glb"),
               ("top5k_tracts_decimated.obj", "tracts", "3001"),
               ("parcelsynth_activations_functionalterms.csv", "parcelsynth", "334"),
               ("structuralconnectivity_metrics_mean.csv", "connectivity", "696²")]
IDLE = "esc leaves · t returns"
STAGES = ("fetch", "strip", "segment", "register", "map")
HELP = [("./main.py <n>...", "the whole pipeline, about 3 min a subject"),
        ("fetch <n>...", "download the scans, then everything after"),
        ("strip <n>...", "skull strip and check, then everything after"),
        ("segment <n>...", "BraTS in docker, then everything after"),
        ("register <n>...", "warp into MNI, then the atlas work, ~40 s"),
        ("map <n>...", "parcel overlap and function terms only, seconds"),
        ("show <n>", "load a subject into the viewer"),
        ("browser <n>", "that subject in a separate browser window"),
        ("cancel", "stop the running job"),
        ("clear", "clear this log"),
        ("help", "this")]


def bar(fraction, n=10):
    filled = round(max(0.0, min(1.0, fraction)) * n)
    return "█" * filled + "░" * (n - filled)


NAMED = {"esc", "enter", "pgup/pgdn"}


def keyed(text):
    # the key carries the brightness, the words behind it stay quiet
    parts = []
    for part in text.split(" · "):
        key, _, rest = part.partition(" ")
        if rest and (len(key) == 1 or key in NAMED):
            parts.append(f"[{FG}]{key}[/] [{DIM}]{rest}[/]")
        else:
            parts.append(f"[{DIM}]{part}[/]")
    return f"[{DIM}] · [/]".join(parts)


def row(left, right, width=30):
    # One list line with the right column flush to the edge.
    room = width - len(right)
    return f"{left[:room]:<{room}}{right}"


class Canvas(ImageWidget, Renderable=Renderable):
    # 3D view: drag to rotate, scroll to zoom.

    dragged = False

    def on_mouse_down(self, event):
        self.capture_mouse()
        self.dragged = False

    def on_mouse_up(self, event):
        self.release_mouse()
        if self.dragged:
            self.app.settle_view()
        else:
            self.app.pick_parcel(event.x, event.y)

    def on_mouse_move(self, event):
        if event.button:
            self.dragged = True
            self.app.queue_view(-event.delta_x * 3, event.delta_y * 3)

    def on_mouse_scroll_down(self, event):
        self.app.queue_view(zoom=0.9)

    def on_mouse_scroll_up(self, event):
        self.app.queue_view(zoom=1.1)

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


class Viewer(Vertical):
    # The modality strip is drawn into the bottom border, so its clicks land here.
    MARGIN, SPACING = 3, 2

    def on_click(self, event):
        # clicks arrive in outer coordinates, so the bottom border is the last row
        if event.y == self.outer_size.height - 1:
            self.app.click_modality(event.x)


class Prompt(Input):
    # Escape gives the keyboard back, up and down walk the history.
    BINDINGS = [("escape", "leave", "leave the prompt")]

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
           border-title-align: left; border-subtitle-align: right;
           border-title-style: bold; border-subtitle-color: #7c6f64;
           background: #282828; padding: 0; }
    #inputs { width: 34; }
    #stats { height: auto; padding: 0 1; }
    #viewer { width: 1fr; }
    #right { width: 46; }
    #grid { layout: grid; grid-size: 2 2; grid-gutter: 0; height: 1fr; }
    #canvas { height: 1fr; }
    .pane { height: 1fr; border: round #504945; border-title-color: #a89984; }
    #readout { height: 1fr; border: round #504945; border-title-color: #a89984;
               padding: 0 1; }
    #shell { height: 14; }
    #shell:focus-within { border: round #fabd2f; }
    #log { height: 1fr; background: #282828; padding: 0 1; }
    #line { height: 1; padding: 0 1; }
    #sigil { width: 2; color: #b8bb26; text-style: bold; }
    #cmd { border: none; height: 1; padding: 0; background: #282828; color: #ebdbb2; }
    #cmd:focus { border: none; background: #32302f; }
    #layers { height: auto; padding: 0 1; }
    #parcels { height: 1fr; background: #282828; }
    ListItem { background: #282828; color: #a89984; padding: 0 1; }
    ListItem.--highlight { background: #32302f; color: #ebdbb2; text-style: bold; }
    #detail { height: 13; border-top: solid #504945; padding: 0 1; }
    #listhead { height: 1; padding: 0 1; color: #ebdbb2; text-style: bold;
                border-top: solid #504945; }
    #inputs, #log, #parcels {
        scrollbar-size-vertical: 1;
        scrollbar-background: #282828; scrollbar-color: #504945;
        scrollbar-background-hover: #282828; scrollbar-color-hover: #7c6f64;
        scrollbar-background-active: #282828; scrollbar-color-active: #a89984; }
    Footer { background: #282828; }
    FooterKey { background: #282828; }
    FooterKey .footer-key--key { color: #cc241d; background: #282828; }
    FooterKey .footer-key--description { color: #a89984; background: #282828; }
    FooterKey.-disabled .footer-key--key { color: #504945; }
    FooterKey.-disabled .footer-key--description { color: #504945; }
    """
    BINDINGS = [
        ("q", "quit", "quit"), ("v", "view", "3d/slices"), ("s", "space", "native/mni"),
        ("p", "palette", "colours"), ("f", "listing", "parcels/functions"),
        ("o", "overlay", "overlay"), ("h", "home", "reset view"),
        ("t", "shell", "prompt"), ("c", "cancel", "cancel run"),
    ]

    def __init__(self):
        super().__init__()
        self.results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        self.subjects = sorted(self.results)
        self.subject = self.subjects[0] if self.subjects else None
        self.mode, self.space, self.palette, self.listing = "3d", "native", "contrast", "parcels"
        self.scene = self.slices = self.proc = self._timer = self._clock = None
        self.hidden_layers, self.subject_rows = set(), {}
        self.history, self.hpos, self.started = [], 0, 0
        self.row_text, self.marked = [], None
        self.pending, self._spin, self._settle = [0.0, 0.0, 1.0], None, None
        self.modality_spans = []
        self._panel_width = self._list_width = 0

    # ---------------------------------------------------------------- compose

    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            with Vertical(id="inputs", classes="box"):
                yield Rows(self.click_subject, id="stats")
            with Viewer(id="viewer", classes="box"):
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
        self.query_one("#inputs").border_title = "inputs"
        self.query_one("#inputs").border_subtitle = "click a subject"
        self.query_one("#shell").border_title = "pipeline"
        self.query_one("#shell").border_subtitle = keyed(IDLE)
        self.query_one("#readout").border_title = "cursor"
        self.log_write(f"[{DIM}]{len(self.results)} subjects from {RESULTS.name}[/]")
        self.log_write(f"[{FAINT}]./main.py <n> runs everything, or start at a stage: "
                       f"{' '.join(STAGES)}. help lists the rest.[/]")
        if not os.environ.get("TERM", "").startswith("xterm-kitty"):
            self.log_write(f"[{WARN}]not kitty: images will render as text blocks[/]")
        if self.subject:
            self.load_subject(self.subject)
            self.call_after_refresh(self.draw)
        else:
            self.log_write(f"[{WARN}]nothing segmented yet, start with ./main.py 36[/]")
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
        self.hidden_layers = set()
        overlap = self.results[subject]["atlas"]
        self.scene = render.Scene(subject, overlap, palette=self.palette)
        self.scene.set_opacity("shell", SHELL_OPACITY)
        self.slices = render.Slices(subject, self.space)
        self.refresh_inputs()
        self.refresh_layers()
        self.refresh_list()
        self.draw()

    def panel_width(self):
        # The panel loses a column whenever its scrollbar appears.
        width = self.query_one("#stats", Rows).size.width
        return width if width > 12 else 30

    def refresh_inputs(self):
        w = self._panel_width = self.panel_width()
        out = [f"[{FG}]subject[/]"]
        self.subject_rows = {}
        for n, s in enumerate(self.subjects, start=1):
            a = self.results[s]["atlas"]
            here = s == self.subject
            self.subject_rows[n] = s
            left = ("› " if here else "  ") + s[10:]
            right = "%3dp %7.0f mL" % (len(a["parcels"]), a["tumour_ml"])
            out.append(f"[{FG if here else DIM}]{row(left, right, w)}[/]")
        out.append(f"[{FAINT}]{'─' * w}[/]")

        d = self.results[self.subject]
        r = d["regions_ml"]
        out.append(f"[{DIM}]tumour[/]")
        out += [f"[{FG}]{row('  ' + name, f'{r[name]} mL', w)}[/]" for name in ("WT", "TC", "ET")]
        in_parcels = f"{d['atlas']['in_parcels_ml']} mL"
        out.append(f"[{FAINT}]{row('  in parcels', in_parcels, w)}[/]")
        dice = f"{d['dice']['WT']:.2f}" if d.get("dice") else "no reference"
        out.append(f"[{FAINT}]{row('  dice WT', dice, w)}[/]")

        out.append(f"[{DIM}]scans[/]")
        for m in MODALITIES:
            ok = (stripped_dir(self.subject) / f"{self.subject}_{m}.nii.gz").exists()
            state = "stripped" if ok else "missing"
            out.append(f"[{FG if ok else FAINT}]  {m}[/]"
                       f"[{OK if ok else BAD}]{state:>{w - len(m) - 2}}[/]")

        out.append(f"[{DIM}]atlas[/]")
        for name, label, note in ATLAS_FILES:
            have = (ATLAS_INPUT / name).exists()
            state = note if have else "missing"
            out.append(f"[{FG if have else FAINT}]  {label}[/]"
                       f"[{FAINT if have else BAD}]{state:>{w - len(label) - 2}}[/]")
        self.query_one("#stats", Rows).update("\n".join(out))

    def click_subject(self, row_index):
        chosen = self.subject_rows.get(row_index)
        if chosen and chosen != self.subject:
            self.load_subject(chosen)

    # ----------------------------------------------------------------- layers

    def refresh_layers(self):
        out = []
        w = max(self.query_one("#layers", Rows).size.width, 30) - 4
        volumes = self.results[self.subject]["labels_ml"]
        for key, name, colour in LAYERS:
            on = key not in self.hidden_layers
            if key == "parcels":
                value = str(len(self.scene.parcel_actors)) if self.scene else ""
            elif key == "focus":
                value = ""
            else:
                value = volumes.get(key.split(":")[-1])
                value = f"{value} mL" if value else ""
            out.append(f"[{colour if on else FAINT}]{'[×]' if on else '[ ]'}[/] "
                       f"[{DIM if on else FAINT}]{row(name, value, w)}[/]")
        self.query_one("#layers", Rows).update("\n".join(out))

    def click_layer(self, row_index):
        if not (0 <= row_index < len(LAYERS) and self.scene):
            return
        key = LAYERS[row_index][0]
        self.hidden_layers ^= {key}
        on = key not in self.hidden_layers
        if key == "parcels":
            self.scene.show_parcels(on)
        elif key == "focus":
            # off means every involved parcel stays at full brightness
            self.isolate_selected() if on else self.scene.unisolate()
        else:
            self.scene.set_visible(key, on)
        self.refresh_layers()
        if self.mode == "3d":
            self.draw()

    def isolate_selected(self):
        index = self.query_one("#parcels", ListView).index or 0
        d = self.results[self.subject]
        if self.listing == "parcels":
            rows = d["atlas"]["parcels"]
            if index < len(rows):
                self.scene.isolate(rows[index]["parcel"])
        else:
            top = d.get("function", [])
            if index < len(top):
                self.scene.isolate(n for n, _, _ in
                                   function.drivers(top[index]["term"], d["atlas"], 7))

    # ------------------------------------------------------------------- list

    def refresh_list(self):
        d = self.results[self.subject]
        listing = self.query_one("#parcels", ListView)
        listing.clear()
        self._list_width = listing.size.width
        w = max(listing.size.width - 3, 34)
        panel = self.query_one("#right")
        if self.listing == "parcels":
            rows = d["atlas"]["parcels"]
            self.query_one("#listhead", Static).update(f"parcels involved · {len(rows)}")
            panel.border_title = self.band(panel, "breakdown", 9, ["f functions"])
            panel.border_subtitle = "click here or in the 3d"
            self.row_text = [row(p["parcel"], f"{bar(p['fraction'])} {p['fraction']:>4.0%}", w - 2)
                             for p in rows]
        else:
            top = d.get("function", [])
            self.query_one("#listhead", Static).update(f"functions at risk · {len(top)}")
            panel.border_title = self.band(panel, "breakdown", 9, ["f parcels"])
            panel.border_subtitle = "meta-analytic, not this patient"
            peak = max((f["score"] for f in top), default=1)
            self.row_text = [row(f["term"], f"{bar(f['score'] / peak)} {f['score']:>5.2f}", w - 2)
                             for f in top]
        for text in self.row_text:
            listing.append(ListItem(Label("  " + text)))
        self.marked = None
        self.refresh_detail(0)

    def mark_row(self, index):
        # only the caret moves, redrawing all 99 rows on every arrow key would crawl
        listing = self.query_one("#parcels", ListView)
        for i in (self.marked, index):
            if i is not None and 0 <= i < min(len(listing.children), len(self.row_text)):
                listing.children[i].children[0].update(
                    ("› " if i == index else "  ") + self.row_text[i])
        self.marked = index

    def refresh_detail(self, index):
        d = self.results[self.subject]
        w = max(self.query_one("#detail", Static).size.width, 30)
        if self.listing == "parcels":
            rows = d["atlas"]["parcels"]
            if not rows or index >= len(rows):
                return
            p = rows[index]
            volume = f"{p['in_tumour_ml']:.2f} of {p['parcel_ml']:.2f} mL"
            lines = [f"[{FG}]{p['long_name'][:w]}[/]",
                     f"[{FAINT}]{p['parcel']} · {p['lobe']} · mostly {p['tissue']}[/]",
                     f"[{DIM}]{row('  in tumour', volume, w)}[/]", ""]
            terms = function.for_parcel(p["parcel"], 6)
            peak = max((v for _, v in terms), default=1)
            lines.append(f"[{FAINT}]parcelsynth terms[/]")
            lines += [f"[{DIM}]{row('  ' + t, f'{bar(v / peak, 8)} {v:>5.2f}', w)}[/]"
                      for t, v in terms]
        else:
            top = d.get("function", [])
            if not top or index >= len(top):
                return
            f = top[index]
            rows = function.drivers(f["term"], d["atlas"])
            lines = [f"[{FG}]{f['term']}[/]",
                     f"[{FAINT}]{row('weighted score', str(f['score']), w)}[/]", "",
                     f"[{FAINT}]built from {len(rows)} parcels, strongest first[/]",
                     f"[{FAINT}]{row('', 'involved     share', w)}[/]"]
            lines += [f"[{DIM}]{row('  ' + name, f'{fraction:>8.0%}     {share:.3f}', w)}[/]"
                      for name, fraction, share in rows[:7]]
        self.query_one("#detail", Static).update("\n".join(lines))

    @on(ListView.Highlighted, "#parcels")
    def highlight(self, event):
        index = event.list_view.index or 0
        self.mark_row(index)
        self.refresh_detail(index)
        if self.mode != "3d" or not self.scene or "focus" in self.hidden_layers:
            return
        self.isolate_selected()
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
            title = f"3d · {self.palette} colours"
            viewer.border_title = self.band(viewer, title, len(title), ["v slices"])
            viewer.styles.border_subtitle_align = "right"
            other = "yba" if self.palette == "contrast" else "contrast"
            viewer.border_subtitle = keyed(
                f"drag rotate · scroll zoom · p {other} colours · h reset")
        else:
            for which in render.Slices.PLANES:
                pane = self.query_one(f"#p-{which}", Pane)
                size = pane.content_size
                pane.image = self.slices.plane(
                    which, (max(size.width * cell.width, 32), max(size.height * cell.height, 32)))
            self.refresh_readout()
            r = self.slices.readout()
            title = f"slices · {r['space']}"
            viewer.border_title = self.band(viewer, title, len(title), ["v 3d"])
            strip, taken = self.modality_strip()
            viewer.styles.border_subtitle_align = "left"
            viewer.border_subtitle = self.band(viewer, strip, taken, self.slice_hints())

    def refresh_readout(self):
        r = self.slices.readout()
        w = max(self.query_one("#readout", Static).size.width, 24)
        place = " ".join(str(v) for v in r["coords"]) if r["coords"] else "—"
        self.query_one("#readout", Static).update("\n".join([
            f"[{FAINT}]{row(r['coord_space'], place, w)}[/]",
            f"[{FAINT}]{row('voxel', ' '.join(str(v) for v in r['voxel']), w)}[/]",
            f"[{FAINT}]{row('intensity', str(r['intensity']), w)}[/]",
            f"[{DIM}]{row('label', r['label'], w)}[/]"]))

    def modality_strip(self):
        # greyed out means the current space does not carry it, mni is T1 only
        r = self.slices.readout()
        out, x = [], Viewer.MARGIN
        self.modality_spans = []
        for i, name in enumerate(MODALITIES, start=1):
            if out:
                out.append(" " * Viewer.SPACING)
                x += Viewer.SPACING
            # only the number carries the state, the way the footer keys do
            label = f"{i} {name}"
            here = name in r["modalities"]
            key = FG if name == r["modality"] else FAINT if here else GONE
            out.append(f"[{key}]{i}[/] [{DIM if here else GONE}]{name}[/]")
            self.modality_spans.append((x, x + len(label)))
            x += len(label)
        return "".join(out), x - Viewer.MARGIN

    def slice_hints(self):
        # each key names what it gives you, not what you are looking at
        core = (f"s {'native' if self.space == 'mni' else 'mni'} · "
                f"o {'hide' if self.slices.overlay else 'show'} overlay")
        return [f"pgup/pgdn slice · {core}", core]

    def band(self, widget, left, taken, choices):
        # one border line, a label at each end, the line itself drawn between
        room = widget.outer_size.width - 6 - taken
        for text in choices:
            if len(text) + 5 <= room:
                return left + f" [{DIM}]{'─' * (room - len(text) - 2)}[/] {keyed(text)}"
        return left

    def click_modality(self, x):
        if self.mode != "slices" or not self.slices:
            return
        here = self.slices.readout()["modalities"]
        for slot, (start, end) in enumerate(self.modality_spans):
            if start <= x < end and MODALITIES[slot] in here:
                self.slices.modality = MODALITIES[slot]
                self.draw()
                return

    def pick_parcel(self, column, line):
        # a parcel clicked in the 3d becomes the selected row in the list
        if self.mode != "3d" or not self.scene:
            return
        cell = get_cell_size()
        name = self.scene.pick(column * cell.width + cell.width // 2,
                               line * cell.height + cell.height // 2)
        if name is None:
            return
        names = [r["parcel"] for r in self.results[self.subject]["atlas"]["parcels"]]
        if self.listing != "parcels":
            self.listing = "parcels"
            self.refresh_list()
            self.call_after_refresh(self.select_row, names.index(name))
        else:
            self.select_row(names.index(name))

    def select_row(self, index):
        self.query_one("#parcels", ListView).index = index

    def queue_view(self, azimuth=0.0, elevation=0.0, zoom=1.0):
        # the terminal sends a move event per pixel; rendering one frame each
        # puts the view seconds behind the mouse, so collect and draw on a clock
        if self.mode != "3d" or not self.scene:
            return
        self.pending[0] += azimuth
        self.pending[1] += elevation
        self.pending[2] *= zoom
        if self._spin is None:
            self._spin = self.set_interval(1 / 30, self.flush_view)
        if self._settle is not None:
            self._settle.stop()
        self._settle = self.set_timer(0.3, self.settle_view)

    def flush_view(self):
        azimuth, elevation, zoom = self.pending
        self.pending = [0.0, 0.0, 1.0]
        if not (azimuth or elevation or zoom != 1.0) or not self.scene:
            return
        self.scene.rotate(azimuth, elevation)
        self.scene.zoom(zoom)
        self.draw()

    def settle_view(self):
        # one full resolution frame once the camera stops
        if self._spin is None:
            return
        self._spin.stop()
        self._spin = None
        self.flush_view()
        if self.scene:
            self.draw()

    def seek(self, which, fx, fy):
        if self.mode == "slices" and self.slices:
            self.slices.click(which, fx, fy)
            self.draw()

    # ---------------------------------------------------------------- actions

    VIEW_ACTIONS = ("view", "space", "palette", "listing", "overlay", "home")

    def check_action(self, action, parameters):
        # grey a key out rather than let it look like it did nothing
        if action in self.VIEW_ACTIONS and not self.scene:
            return None
        if action in ("palette", "home"):
            return self.mode == "3d" or None
        if action in ("space", "overlay"):
            return self.mode == "slices" or None
        if action == "cancel":
            return self.proc is not None or None
        return True

    def action_view(self):
        self.mode = "slices" if self.mode == "3d" else "3d"
        self.query_one("#canvas").display = self.mode == "3d"
        self.query_one("#grid").display = self.mode == "slices"
        self.refresh_bindings()
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
        self.query_one("#cmd", Prompt).value = ""
        self.log_write(f"[{OK}]$[/] [{FG}]{line}[/]")
        if not line:
            return
        self.history.append(line)
        self.hpos = len(self.history)
        self.execute(line)

    def execute(self, line):
        parts = line.split()
        name = parts[0].lstrip("./")
        args = parts[1:]

        # a stage name is shorthand for main.py --from that stage
        stage = "fetch"
        if name in STAGES or name in ("main.py", "main", "run"):
            if name in STAGES:
                stage = name
            if "--from" in args:
                cut = args.index("--from")
                stage = args[cut + 1] if cut + 1 < len(args) else ""
                args = args[:cut] + args[cut + 2:]
            if stage not in STAGES:
                self.log_write(f"[{BAD}]stages are {', '.join(STAGES)}[/]")
                return

        numbers = []
        for arg in args:
            if not arg.isdigit():
                self.log_write(f"[{BAD}]{arg}: expected a subject number[/]")
                return
            numbers.append(int(arg))

        if name in STAGES or name in ("main.py", "main", "run"):
            if not numbers:
                self.log_write(f"[{BAD}]usage: {name} <subject numbers>[/]")
            elif self.proc:
                self.log_write(f"[{WARN}]a job is already running, cancel it first[/]")
            else:
                self.run_pipeline(numbers, stage)
        elif name in ("show", "load"):
            self.show_subject(numbers)
        elif name in ("browser", "view.py", "view"):
            self.open_browser(numbers)
        elif name in ("cancel", "stop"):
            self.action_cancel()
        elif name == "clear":
            self.query_one("#log", RichLog).clear()
        elif name in ("quit", "exit"):
            self.exit()
        elif name == "help":
            for command, note in HELP:
                self.log_write(f"[{DIM}]{command:<20}[/][{FAINT}]{note}[/]")
        else:
            self.log_write(f"[{BAD}]{name}: not a command. try help.[/]")

    def show_subject(self, numbers):
        if not numbers:
            self.log_write(f"[{BAD}]usage: show <subject number>[/]")
            return
        wanted = subject_id(numbers[0])
        if wanted not in self.results:
            self.log_write(f"[{BAD}]{wanted} has no results yet[/]")
            return
        self.load_subject(wanted)

    def open_browser(self, numbers):
        wanted = subject_id(numbers[0]) if numbers else self.subject
        if wanted not in self.results:
            self.log_write(f"[{BAD}]{wanted} has no results yet[/]")
            return
        # say where it went, it draws in a window that is not this one
        self.log_write(f"[{DIM}]{wanted} is opening in a separate browser window[/]")
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
        self.query_one("#shell").border_subtitle = keyed(note)

    def tick(self):
        self.busy(f"running {time.time() - self.started:.0f}s · c cancels")

    @work(exclusive=True)
    async def run_pipeline(self, numbers, stage="fetch"):
        self.started = time.time()
        self._clock = self.set_interval(1.0, self.tick)
        self.tick()
        # --no-view because this panel already has the viewer next to it
        call = ["main.py", *[str(n) for n in numbers], "--no-view"]
        if stage != "fetch":
            call += ["--from", stage]
        self.log_write(f"[{FAINT}]{' '.join(call)}[/]")
        # rich draws a live spinner that becomes escape-code debris in a log widget
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, *call, cwd=str(ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        self.refresh_bindings()
        async for raw in self.proc.stdout:
            text = ANSI.sub("", raw.decode(errors="replace")).rstrip()
            if text:
                self.log_write(f"[{DIM}]{text}[/]")
        code = await self.proc.wait()
        elapsed = time.time() - self.started
        self.proc = None
        self._clock.stop()
        self.busy(IDLE)
        self.refresh_bindings()
        if code < 0:
            return
        if code != 0:
            self.log_write(f"[{BAD}]exited {code}[/]")
            return
        self.results = json.loads(RESULTS.read_text())
        self.subjects = sorted(self.results)
        self.log_write(f"[{OK}]done in {elapsed:.0f}s[/]")
        self.load_subject(subject_id(numbers[-1]))


if __name__ == "__main__":
    Console().run()
