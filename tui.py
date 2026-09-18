#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Terminal UI for the segmentation pipeline. Needs kitty for the graphics protocol.

import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
import time

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Input, Label, ListItem, ListView, RichLog, Static
from textual_image._terminal import get_cell_size
from textual_image.renderable import Image as Renderable
from textual_image.widget import Image as ImageWidget

import atlas
import compare
import connect
import function
import render
import segment
from paths import (ATLAS_INPUT, LABELS, MODALITIES, REGIONS, RESULTS, ROOT,
                   STAGES, out_dir, raw_dir, stage_file, stripped_dir, subject_id)

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")
MARKER = re.compile(r"^\* (\w+)$")
SUBJECT = re.compile(r"^UPENN-GBM-\d{5}_\d\d$")

# gruvbox dark, the same values btop reads out of gruvbox_dark_v2.theme
FG, DIM, FAINT = "#ebdbb2", "#a89984", "#7c6f64"
OK, WARN, BAD, GONE = "#b8bb26", "#fabd2f", "#fb4934", "#504945"
SHELL_OPACITY = 0.15

TISSUE = {"NETC": "#f2c14e", "SNFH": "#3fa7d6", "ET": "#e4572e", "RC": "#8e7dbe",
          "WT": "#d6c8af"}
# the whole tumor as one shape, with its three parts under it
PARTS = ("T:SNFH", "T:ET", "T:NETC", "T:RC")
LAYERS = [("shell", "brain surface", "#a89984", 0),
          ("--a", "", "", 0),
          ("T:margin", "margin 2 cm", "#b16286", 0),
          ("T:WT", "tumor", TISSUE["WT"], 0),
          ("T:SNFH", "SNFH edema", TISSUE["SNFH"], 1),
          ("T:ET", "ET enhancing", TISSUE["ET"], 1),
          ("T:NETC", "NETC necrosis", TISSUE["NETC"], 1),
          ("T:RC", "RC cavity", TISSUE["RC"], 1),
          ("--b", "", "", 0),
          ("tracts", "tracts through the tumor", "#c67c48", 0),
          ("tracts:near", "tracts clipping a parcel", "#76a096", 0),
          ("--c", "", "", 0),
          ("C:both", "both timepoints", "#ebdbb2", 0),
          ("C:new", "new in the later", "#fb4934", 0),
          ("C:gone", "gone since the first", "#8ec07c", 0)]
# WT, TC and ET are unions of the tissue classes, so they are deliberately left
# uncolored: any class color on them reads as "this row is that tissue"
ATLAS_FILES = [("YBA_696.nii", "volume", "696"),
               ("YBA_696.glb", "meshes", "glb"),
               ("top5k_tracts_decimated.obj", "tracts", "3001"),
               ("parcelsynth_activations_functionalterms.csv", "parcelsynth", "334"),
               ("structuralconnectivity_metrics_mean.csv", "connectivity", "696²")]
VIEWERS = ("3d", "slices", "graph")
VIEWS = [("a", "parcels"), ("b", "functions"), ("c", "connectivity")]
ABOUT = {
    "parcels": "Every parcel the tumor reaches, ranked by how much of that parcel "
               "is inside it.",
    "functions": "Each involved parcel's Parcelsynth profile, weighted by how much of "
                 "that parcel the tumor holds, then summed. Pooled across published "
                 "studies, not this patient.",
    "connectivity": "For each parcel the tumor missed, what it connects to that the "
                    "tumor took, weighted by how much of each is gone. A healthy "
                    "population mean, not this patient.",
}
IDLE = "esc leaves · t returns"
STAGES = ("fetch", "strip", "segment", "register", "map")
HELP = [("./main.py <n>...", "the whole pipeline, about 3 min a subject"),
        ("--timepoint 2", "on any of these, the follow-up scan"),
        ("fetch <n>...", "download the scans, then everything after"),
        ("strip <n>...", "skull strip and check, then everything after"),
        ("segment <n>...", "BraTS in docker, then everything after"),
        ("register <n>...", "warp into MNI, then the atlas work, ~40 s"),
        ("map <n>...", "parcel overlap and function terms only, seconds"),
        ("show <n>", "load a subject into the viewer"),
        ("browser <n>", "that subject in a separate browser window"),
        ("save", "the viewer as a png, next to the outputs"),
        ("cancel", "stop the running job"),
        ("clear", "clear this log"),
        ("help", "this")]


def bar(fraction, n=10):
    filled = round(max(0.0, min(1.0, fraction)) * n)
    return "█" * filled + "░" * (n - filled)


NAMED = {"esc", "enter"}


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


def ago(when):
    # "4m", "2d", coarse on purpose, this is freshness not a timestamp
    gap = max(time.time() - when, 0)
    for size, mark in ((86400, "d"), (3600, "h"), (60, "m")):
        if gap >= size:
            return f"{int(gap // size)}{mark} ago"
    return f"{int(gap)}s ago"


def spell(seconds):
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s" if seconds >= 60 else f"{seconds}s"


def distribution(overlap):
    # Lobes under the hemisphere they belong to, and only when both are involved.
    side_of = {p["name"]: p["hemisphere"] for p in atlas.parcels()}
    tree = {}
    for r in overlap["parcels"]:
        side = tree.setdefault(side_of.get(r["parcel"], "?"), {})
        side[r["lobe"]] = side.get(r["lobe"], 0) + r["in_tumor_ml"]
    order = sorted(tree.items(), key=lambda kv: -sum(kv[1].values()))
    if not order:
        return []
    if len(order) == 1:
        return [(0, name.lower(), round(ml, 2))
                for name, ml in sorted(order[0][1].items(), key=lambda kv: -kv[1])]
    out = []
    for side, lobes in order:
        out.append((0, side.lower(), round(sum(lobes.values()), 2)))
        out += [(1, name.lower(), round(ml, 2))
                for name, ml in sorted(lobes.items(), key=lambda kv: -kv[1])]
    return out


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


class Board(ImageWidget, Renderable=Renderable):
    # The node chart: drag to pan, scroll to zoom, click a node to pick it.
    dragged = False

    def on_mouse_down(self, event):
        self.capture_mouse()
        self.dragged = False

    def on_mouse_up(self, event):
        self.release_mouse()
        if not self.dragged:
            self.app.pick_node(event.x, event.y)

    def on_mouse_move(self, event):
        if event.button:
            self.dragged = True
            cell = get_cell_size()
            self.app.queue_chart(pan=(event.delta_x * cell.width,
                                      event.delta_y * cell.height))

    def on_mouse_scroll_down(self, event):
        if not self.app.scroll_shelf(event.x, event.y, 3):
            self.app.queue_chart(zoom=0.85)

    def on_mouse_scroll_up(self, event):
        if not self.app.scroll_shelf(event.x, event.y, -3):
            self.app.queue_chart(zoom=1.18)

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


class Bordered(VerticalScroll):
    # A strip is drawn into the bottom border, so its clicks land here.
    MARGIN, SPACING = 3, 2

    def on_click(self, event):
        # clicks arrive in outer coordinates, so the bottom border is the last row
        if event.y == self.outer_size.height - 1:
            self.app.click_strip(self.id, event.x)


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
    #inputs { width: 34; scrollbar-gutter: stable; }
    #stats { height: auto; padding: 0 1; }
    #viewer { width: 1fr; }
    #right { width: 46; }
    #grid { layout: grid; grid-size: 2 2; grid-gutter: 0; height: 1fr; }
    #canvas { height: 1fr; }
    #board { height: 1fr; }
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
    #parcels { height: 1fr; min-height: 5; background: #282828; }
    ListItem { background: #282828; color: #a89984; padding: 0 1; }
    ListItem.--highlight { background: #32302f; color: #ebdbb2; text-style: bold; }
    #detail { height: 13; border-top: solid #504945; padding: 0 1;
              scrollbar-gutter: stable; scrollbar-size-vertical: 1; scrollbar-background: #282828;
              scrollbar-color: #504945; scrollbar-color-hover: #7c6f64;
              scrollbar-color-active: #a89984; }
    #detailtext { height: auto; }
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
        ("q", "quit", "quit"), ("v", "view", "3d/slices"), ("s", "space", "space/layout"),
        ("p", "palette", "colors"), ("t", "shell", "prompt"),
        ("x", "cancel", "cancel run"),
        Binding("a", "show_list('parcels')", "parcels", show=False),
        Binding("b", "show_list('functions')", "functions", show=False),
        Binding("c", "show_list('connectivity')", "connectivity", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        self.subjects = sorted(self.results)
        self.subject = self.subjects[0] if self.subjects else None
        self.mode, self.space, self.palette, self.listing = "3d", "native", "contrast", "parcels"
        self.scene = self.slices = self.chart = None
        self.proc = self._timer = self._clock = None
        self.hidden_layers, self.subject_rows = {"T:margin",
                                          "C:both", "C:new", "C:gone"}, {}
        self.layer_rows = list(LAYERS)
        self.history, self.hpos, self.started = [], 0, 0
        self.stage, self.watching = None, None
        self.row_text, self.marked, self.at_risk = [], None, []
        self.parcel_rows = []
        self.pending, self._spin, self._settle = [0.0, 0.0, 1.0], None, None
        self.waiting = [1.0, 0.0, 0.0]
        self.strip_spans = {}
        self.strip_spans = {}
        self._panel_width = self._list_width = self._layers_width = 0

    # ---------------------------------------------------------------- compose

    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            with VerticalScroll(id="inputs", classes="box"):
                yield Rows(self.click_subject, id="stats")
            with Bordered(id="viewer", classes="box"):
                yield Canvas(id="canvas")
                yield Board(id="board")
                with Grid(id="grid"):
                    yield Pane("axial", id="p-axial", classes="pane")
                    yield Pane("coronal", id="p-coronal", classes="pane")
                    yield Pane("sagittal", id="p-sagittal", classes="pane")
                    yield Static("", id="readout")
            with Bordered(id="right", classes="box"):
                yield Rows(self.click_layer, id="layers")
                yield Static("", id="listhead")
                yield ListView(id="parcels")
                with VerticalScroll(id="detail"):
                    yield Static("", id="detailtext")
        with Vertical(id="shell", classes="box"):
            yield RichLog(id="log", wrap=True, markup=True, highlight=False)
            with Horizontal(id="line"):
                yield Static("$", id="sigil")
                yield Prompt(id="cmd", placeholder="./main.py 36")
        yield Footer()

    def on_mount(self):
        self.query_one("#grid").display = False
        self.query_one("#board").display = False
        self.query_one("#inputs").border_title = "inputs"
        self.query_one("#inputs").border_subtitle = "click a subject"
        self.query_one("#shell").border_title = "pipeline"
        self.query_one("#shell").border_subtitle = keyed(IDLE)
        self.query_one("#readout").border_title = "cursor"
        self.query_one("#right").border_title = "breakdown"
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
        # results.json can outlive the files it describes, and that is not fatal
        missing = [name for name in ("seg.nii.gz", "seg_in_mni.nii.gz")
                   if not (out_dir(subject) / name).exists()]
        if missing:
            self.log_write(f"[{BAD}]{subject}: {', '.join(missing)} is gone, "
                           f"rerun with ./main.py {int(subject[10:15])}[/]")
            return False
        self.subject, self.watching = subject, None
        self.hidden_layers = {"T:margin", "C:both", "C:new", "C:gone"}
        overlap = self.results[subject]["atlas"]
        self.scene = render.Scene(subject, overlap, palette=self.palette)
        self.scene.set_opacity("shell", SHELL_OPACITY)
        self.slices, self.chart = render.Slices(subject, self.space), None
        self.apply_layers()
        self.refresh_inputs()
        self.refresh_layers()
        self.refresh_list()
        self.draw()
        return True

    def panel_width(self):
        # The panel loses a column whenever its scrollbar appears.
        width = self.query_one("#stats", Rows).size.width
        return width if width > 12 else 30

    def refresh_inputs(self):
        w = self._panel_width = self.panel_width()
        out = [f"[{FG}]subject[/]"]
        self.subject_rows = {}
        counts = {}
        for name in self.subjects:
            counts[name[10:15]] = counts.get(name[10:15], 0) + 1
        drawn = set()
        for n, name in enumerate(self.subjects, start=1):
            a = self.results[name]["atlas"]
            here = name == self.subject
            self.subject_rows[n] = name
            patient = name[10:15]
            if counts[patient] < 2:
                tie = " "
            elif patient not in drawn:
                tie = "\u250c"
            else:
                tie = "\u2514" if name == [s for s in self.subjects
                                           if s[10:15] == patient][-1] else "\u2502"
            drawn.add(patient)
            left = ("\u203a " if here else "  ") + tie + " " + name[10:]
            right = "%3dp %7.0f mL" % (len(a["parcels"]), a["tumor_ml"])
            out.append(f"[{FG if here else DIM}]{row(left, right, w)}[/]")
        def rule(title):
            return f"[{FAINT}]\u2500\u2500 [/][{FG}]{title}[/][{FAINT}] {'\u2500' * (w - 4 - len(title))}[/]"

        d = self.results[self.subject]
        r = d["regions_ml"]
        out.append(rule("stats"))
        out.append(f"[{DIM}]tumor[/]")
        for name in ("WT", "TC", "ET", "RC"):
            if not r.get(name):
                continue
            # each token names exactly one class, so it can carry that color
            parts = [LABELS[v] for v in REGIONS[name]]
            made = "" if len(parts) == 1 else " " + "+".join(parts)
            painted = "" if len(parts) == 1 else " " + f"[{FAINT}]+[/]".join(
                f"[{TISSUE[t]}]{t}[/]" for t in parts)
            out.append(f"[{FG}]  {name}[/]{painted}"
                       f"[{DIM}]{f'{r[name]} mL':>{w - 4 - len(made)}}[/]")
        in_parcels = f"{d['atlas']['in_parcels_ml']} mL"
        out.append(f"[{FAINT}]  in parcels[/]"
                   f"[{DIM}]{in_parcels:>{w - 12}}[/]")
        # nested under the line they break down, in the same unit as it
        out += [f"[{FAINT}]{row('    ' + '  ' * deep + name, f'{value} mL', w)}[/]"
                for deep, name, value in distribution(d["atlas"])]
        group = [s for s in self.subjects if s[10:15] == self.subject[10:15]]
        if len(group) > 1:
            out.append(f"[{DIM}]changes[/]")
            for before, after in zip(group, group[1:]):
                change = (self.results[after]["regions_ml"]["WT"]
                          - self.results[before]["regions_ml"]["WT"])
                label = f"  _{before[16:]} \u2192 _{after[16:]} WT"
                out.append(f"[{WARN if change > 0 else FAINT}]"
                           f"{row(label, f'{change:+.1f} mL', w)}[/]")
                was = {r["parcel"] for r in self.results[before]["atlas"]["parcels"]}
                now = {r["parcel"] for r in self.results[after]["atlas"]["parcels"]}
                out.append(f"[{WARN if now - was else FAINT}]"
                           f"{row('  newly involved', str(len(now - was)), w)}[/]")
                out.append(f"[{FAINT}]{row('  no longer', str(len(was - now)), w)}[/]")
                same = compare.against(before, after)
                out.append(f"[{FAINT}]{row('  dice', f"{same['dice']:.2f}", w)}[/]")
                label = f"  of _{after[16:]} inside _{before[16:]}"
                out.append(f"[{FAINT}]{row(label, f"{same['inside']:.0%}", w)}[/]")

        run = d.get("run", {})
        out.append(f"[{DIM}]run[/]")
        if "algorithm" in run:
            made = f"  {run['algorithm']} {run.get('device', '')}".rstrip()
            out.append(f"[{FAINT}]{row(made, spell(run['seconds']) if 'seconds' in run else '', w)}[/]")
        else:
            out.append(f"[{FAINT}]{row('  segmenter', 'not recorded', w)}[/]")
        out.append(f"[{FAINT}]{row('  dice WT', self.dice(d), w)}[/]")
        if "kept" in run:
            out.append(f"[{FAINT}]{row('  aligned on', f"{run['kept']:.0%}", w)}[/]")

        out.append(rule("checks"))
        out.append(f"[{DIM}]stages[/]")
        for name, note, gap, trouble in self.stage_checks(d, run):
            mark, tone = ("\u2713", FAINT) if trouble is None else ("\u2717", BAD)
            out.append(f"[{tone}]  {mark}[/] [{FAINT}]{name}[/]"
                       f"[{tone if trouble else FAINT}]{note:>{w - len(name) - 4}}[/]")
            if trouble:
                out.append(f"[{BAD}]{row('    \u2514 ' + trouble, '', w)}[/]")

        out.append(f"[{DIM}]atlas[/]")
        for name, label, note in ATLAS_FILES:
            have = (ATLAS_INPUT / name).exists()
            mark, tone = ("\u2713", FAINT) if have else ("\u2717", BAD)
            out.append(f"[{tone}]  {mark}[/] [{FAINT}]{label}[/]"
                       f"[{FAINT if have else BAD}]{(note if have else "missing"):>{w - len(label) - 4}}[/]")

        self.query_one("#stats", Rows).update("\n".join(out))

    def stage_checks(self, d, run):
        # name, what it says, and what is wrong if anything
        times, stamps, out = run.get("times", {}), {}, []
        for stage in STAGES:
            path = stage_file(self.subject, stage)
            stamps[stage] = (run.get("mapped") if path is None
                             else path.stat().st_mtime if path.exists() else None)
            trouble = self.gap(stage, d)
            if stamps[stage] is None and stage != "map":
                out.append((stage, "never run", None, trouble or "no output"))
                continue
            done = [t for t in stamps.values() if t]
            stale = len(done) > 1 and stamps[stage] and stamps[stage] < max(done[:-1])
            when = ago(stamps[stage]) if stamps[stage] else "unknown"
            note = f"{spell(round(times[stage]))} \u00b7 {when}" if stage in times else when
            out.append((stage, "stale, " + when if stale else note, stamps[stage],
                        trouble or ("left behind by a rerun" if stale else None)))
        return out

    def gap(self, stage, d):
        # the specific thing missing, which is what the red sub row says
        if stage == "fetch":
            short = [m for m in MODALITIES
                     if not (raw_dir(self.subject) / f"{self.subject}_{m}.nii.gz").exists()]
            return ", ".join(short) + " not downloaded" if short else None
        if stage == "strip":
            short = [m for m in MODALITIES
                     if not (stripped_dir(self.subject) / f"{self.subject}_{m}.nii.gz").exists()]
            if short:
                return ", ".join(short) + " not stripped"
            found = out_dir(self.subject) / "verify.txt"
            if found.exists() and found.read_text().strip():
                return found.read_text().strip().splitlines()[0]
            return None
        if stage == "map" and "atlas" not in d:
            return "no parcel overlap"
        return None

    def dice(self, d):
        return f"{d['dice']['WT']:.2f}" if d.get("dice") else "no reference"

    def pair(self):
        # the patient's two timepoints, earliest first, or nothing
        group = [s for s in self.subjects if s[10:15] == self.subject[10:15]]
        return tuple(group[:2]) if len(group) > 1 else None

    def sibling(self, subject):
        # the same patient at the other timepoint, if it has been run
        other = next((s for s in self.subjects
                      if s[10:15] == subject[10:15] and s != subject), None)
        return other

    def click_subject(self, row_index):
        chosen = self.subject_rows.get(row_index)
        if chosen and chosen != self.subject:
            self.load_subject(chosen)

    # ----------------------------------------------------------------- layers

    def refresh_layers(self):
        out = []
        self._layers_width = self.query_one("#layers", Rows).size.width
        w = max(self._layers_width, 30) - 4
        volumes = self.results[self.subject]["labels_ml"]
        regions = self.results[self.subject]["regions_ml"]
        pair = self.pair()
        self.layer_rows = [r for r in LAYERS
                           if (not r[0].startswith("T:") or r[0] in ("T:WT", "T:margin")
                               or r[0][2:] in volumes)
                           and (pair or not (r[0].startswith("C:") or r[0] == "--c"))]
        for key, name, color, deep in self.layer_rows:
            if key.startswith("--"):
                out.append(f"[{FAINT}]{'\u2500' * (w + 4)}[/]")
                continue
            on = key not in self.hidden_layers
            if key.startswith("tracts"):
                value = str(self.scene.tract_counts.get(key, "")) if self.scene else ""
            elif key.startswith("C:"):
                value = f"{compare.against(*pair)[key[2:]]} mL"
            elif key == "T:margin":
                grown = self.results[self.subject].get("margin_ml")
                value = f"{grown} mL" if grown else "rerun map"
            elif key == "T:WT":
                value = f"{regions['WT']} mL"
            else:
                value = volumes.get(key.split(":")[-1])
                value = f"{value} mL" if value else ""
            out.append("  " * deep + f"[{color if on else FAINT}]"
                       f"{'[\u00d7]' if on else '[ ]'}[/] "
                       f"[{DIM if on else FAINT}]{row(name, value, w - 2 * deep)}[/]")
        self.query_one("#layers", Rows).update("\n".join(out))

    def shown_parcels(self):
        # the parcels the layers are actually drawing, which is what the list shows
        rows = self.results[self.subject]["atlas"]["parcels"]
        change = tuple(k[2:] for k, *_ in LAYERS
                       if k.startswith("C:") and k not in self.hidden_layers)
        if change and self.pair():
            touched = compare.parcels_in(*self.pair(), change)
            return [r for r in rows if r["parcel"] in touched]
        # only the classes this subject actually has; RC is usually absent
        present = self.results[self.subject]["labels_ml"]
        classes = {name for name in LABELS.values()
                   if name in present and "T:" + name not in self.hidden_layers}
        if not classes:
            return rows if "T:WT" not in self.hidden_layers else []
        return [r for r in rows if r["tissue"] in classes]

    def apply_layers(self):
        # the one place that turns the checkbox states into what is drawn
        if not self.scene:
            return
        for key, *_ in LAYERS:
            if key.startswith("--"):
                continue
            on = key not in self.hidden_layers
            # the combined shape stands in for the parts when none of them is on
            if key == "T:WT":
                on = on and not self.split(key)
            if key in self.scene.actors:
                self.scene.set_visible(key, on)
        self.sync_overlay()

    def split(self, parent):
        # the children of a parent row that are on and have something to draw
        kin = PARTS
        return [k for k in kin
                if k not in self.hidden_layers and k in self.scene.actors]

    def click_layer(self, row_index):
        if not (0 <= row_index < len(self.layer_rows) and self.scene):
            return
        key = self.layer_rows[row_index][0]
        if key.startswith("--"):
            return
        tumor = {"T:WT", *PARTS}
        change = {k for k, *_ in LAYERS if k.startswith("C:")}
        if key == "T:WT":
            # the parent carries its parts with it
            self.hidden_layers = (self.hidden_layers - tumor if key in self.hidden_layers
                                  else self.hidden_layers | tumor)
        else:
            self.hidden_layers ^= {key}
            if key in PARTS and key not in self.hidden_layers:
                self.hidden_layers.discard("T:WT")
        # these two draw the same tissue, so only one of them at a time. nothing
        # else is a rival: the margin, the tracts and the shell sit over any of it
        if key not in self.hidden_layers:
            if key in tumor:
                self.hidden_layers |= change
            elif key in change:
                self.hidden_layers |= tumor
        if key == "T:margin" and key not in self.hidden_layers:
            self.scene.ensure_margin()
        if key.startswith("C:") and key not in self.hidden_layers:
            self.scene.ensure_compare(*self.pair())
        self.apply_layers()
        self.refresh_layers()
        self.refresh_list()
        self.draw()

    def sync_overlay(self):
        # the same checkbox hides a class in the slices as in the 3d
        if not self.slices:
            return
        self.slices.hidden = {value for value, name in LABELS.items()
                              if "T:" + name in self.hidden_layers}
        whole = "T:WT" not in self.hidden_layers and not self.split("T:WT")
        self.slices.whole = render.TISSUE[0] if whole else None
        self.slices.show_margin = "T:margin" not in self.hidden_layers
        self.slices.compare = {}
        pair = self.pair()
        wanted = [k[2:] for k, *_ in LAYERS
                  if k.startswith("C:") and k not in self.hidden_layers]
        if pair and wanted:
            # in mni the pieces are already on the grid; in native they are
            # brought back through this subject's own inverse warp
            source = (compare.masks(*pair) if self.space == "mni"
                      else compare.native(self.subject, *pair))
            for name in wanted:
                if name in source:
                    self.slices.compare[name] = self.slices.canon_mask(source[name])
        self.slices.involved = {}
        if self.slices.parcels is None:
            return
        # the same parcels the 3d draws, in the same colors
        order = {p["name"]: i for i, p in enumerate(atlas.parcels())}
        palette = atlas.colors(self.palette)
        for name in self.selected_parcels():
            if name in order:
                self.slices.involved[order[name] + 1] = palette[order[name]]

    def picked(self, index=None):
        # the first row is the empty pick, so the data starts one below the list
        if index is None:
            index = self.query_one("#parcels", ListView).index or 0
        return index - 1

    def selected_parcels(self):
        # the parcels the current row is about: one, or the few that define it
        index = self.picked()
        if index < 0:
            return []
        d = self.results[self.subject]
        if self.listing == "parcels":
            rows = self.parcel_rows
            return [rows[index]["parcel"]] if index < len(rows) else []
        if self.listing == "connectivity":
            if index >= len(self.at_risk):
                return []
            name = self.at_risk[index]["parcel"]
            return [name] + [n for n, _, _ in connect.partners(name, d["atlas"], 12)]
        top = d.get("function", [])
        if index >= len(top):
            return []
        return [n for n, _, _ in function.drivers(top[index]["term"], d["atlas"], 7)]

    def isolate_selected(self):
        names = self.selected_parcels()
        self.scene.only(names)
        # in the connectivity view the first one is not in the tumor, so it has
        # no mesh in the scene until it is asked for
        self.scene.spotlight(names[0] if self.listing == "connectivity" and names else None)

    def refresh_list(self):
        d = self.results[self.subject]
        listing = self.query_one("#parcels", ListView)
        listing.clear()
        self._list_width = listing.size.width
        w = max(listing.size.width - 5, 32)
        panel = self.query_one("#right")
        chips = self.chips()
        head = self.query_one("#listhead", Static)

        if self.listing == "parcels":
            rows = self.parcel_rows = self.shown_parcels()
            head.update(f"parcels shown \u00b7 {len(rows)}")
            body = [chips.get(p["parcel"], " ") + " " +
                    row(p["parcel"], f"{bar(p['fraction'])} {p['fraction']:>4.0%}", w - 2)
                    for p in rows]
        elif self.listing == "functions":
            rows = d.get("function", [])
            head.update(f"functions at risk \u00b7 {len(rows)}")
            peak = max((f["score"] for f in rows), default=1)
            body = [row("  " + f["term"], f"{bar(f['score'] / peak)} {f['score']:>5.2f}", w)
                    for f in rows]
        else:
            rows = self.at_risk = connect.at_risk(d["atlas"])
            head.update(f"connectivity at risk \u00b7 {len(rows)}")
            peak = max((c["share"] for c in rows), default=1)
            body = [chips.get(c["parcel"], " ") + " " +
                    row(c["parcel"], f"{bar(c['share'] / peak)} {c['share']:>5.0%}", w - 2)
                    for c in rows]

        self.row_text = [f"[{FAINT}]nothing selected[/]"] + body
        panel.border_title = f"breakdown \u00b7 {self.listing}"
        panel.border_subtitle = self.strip("right", VIEWS, self.listing)[0]
        panel.styles.border_subtitle_align = "left"
        for text in self.row_text:
            listing.append(ListItem(Label("  " + text)))
        self.marked = None
        # park the caret on the empty pick, so the list agrees with the viewers
        listing.index = 0
        self.refresh_detail(0)

    def mark_row(self, index):
        # only the caret moves, redrawing all 99 rows on every arrow key would crawl
        listing = self.query_one("#parcels", ListView)
        for i in (self.marked, index):
            if i is not None and 0 <= i < min(len(listing.children), len(self.row_text)):
                listing.children[i].children[0].update(
                    ("› " if i == index else "  ") + self.row_text[i])
        self.marked = index

    def chips(self):
        # same color the parcel has in the 3d, so a row and a shape match up
        palette = atlas.colors(self.palette)
        return {p["name"]: "[rgb(%d,%d,%d)]█[/]" % tuple(palette[i])
                for i, p in enumerate(atlas.parcels())}

    def refresh_detail(self, index):
        d = self.results[self.subject]
        w = max(self.query_one("#detailtext", Static).size.width, 30)
        text = self.query_one("#detailtext", Static)
        about = [""] + [f"[{FAINT}]{line}[/]"
                        for line in textwrap.wrap(ABOUT[self.listing], w)]
        index = self.picked(index)
        if index < 0:
            text.update("\n".join(about[1:]))
            return

        if self.listing == "parcels":
            rows = self.parcel_rows
            if index >= len(rows):
                return
            p = rows[index]
            volume = f"{p['in_tumor_ml']:.2f} of {p['parcel_ml']:.2f} mL"
            lines = [f"[{FG}]{p['long_name'][:w]}[/]",
                     f"[{FAINT}]{p['parcel']} \u00b7 {p['lobe']} \u00b7 mostly {p['tissue']}[/]",
                     f"[{DIM}]{row('  in tumor', volume, w)}[/]", ""]
            terms = function.for_parcel(p["parcel"], None)
            peak = max((v for _, v in terms), default=1)
            lines.append(f"[{FAINT}]parcelsynth terms[/]")
            lines += [f"[{DIM}]{row('  ' + t, f'{bar(v / peak, 8)} {v:>5.2f}', w)}[/]"
                      for t, v in terms]
        elif self.listing == "connectivity":
            if index >= len(self.at_risk):
                return
            c = self.at_risk[index]
            info = self.parcel_info().get(c["parcel"], {})
            rows = connect.partners(c["parcel"], d["atlas"])
            lines = [f"[{FG}]{info.get('long_name', c['parcel'])[:w]}[/]",
                     f"[{FAINT}]{c['parcel']} \u00b7 {info.get('lobe', '')} \u00b7 "
                     f"not in the tumor[/]",
                     f"[{DIM}]{row('  at risk', f"{c['share']:.0%}", w)}[/]", "",
                     f"[{FAINT}]{row(f'{len(rows)} parcels', 'involved    of loss', w)}[/]"]
            lines += [f"[{DIM}]{row('  ' + name, f'{fraction:>8.0%}    {share:>6.1%}', w)}[/]"
                      for name, fraction, share in rows]
        else:
            top = d.get("function", [])
            if index >= len(top):
                return
            f = top[index]
            rows = function.drivers(f["term"], d["atlas"])
            lines = [f"[{FG}]{f['term']}[/]",
                     f"[{FAINT}]{row('weighted score', str(f['score']), w)}[/]", "",
                     f"[{FAINT}]{row(f'{len(rows)} parcels', 'involved     share', w)}[/]"]
            lines += [f"[{DIM}]{row('  ' + name, f'{fraction:>8.0%}     {share:.3f}', w)}[/]"
                      for name, fraction, share in rows]
        text.update("\n".join(lines + about))

    @on(ListView.Highlighted, "#parcels")
    def highlight(self, event):
        index = event.list_view.index or 0
        self.mark_row(index)
        self.refresh_detail(index)
        # a different parcel is a different list of tracts, so the picked one goes
        if self.chart is not None:
            self.chart.tract = None
        if self.scene:
            self.scene.show_tract(None)
        self.focus_row(index)
        self.draw()

    def focus_row(self, index=None):
        # the picked row stays picked when the viewer changes under it
        if index is None:
            index = self.query_one("#parcels", ListView).index or 0
        self.sync_overlay()
        if self.mode == "3d" and self.scene:
            self.isolate_selected()
        elif self.mode == "slices":
            self.seek_parcel(index)
        elif self.mode == "graph" and self.chart is not None:
            self.light_node()

    def light_node(self):
        # a parcel row and a connectivity row are each about one parcel, so one
        # node lights. a function is a weighted sum over many, so all of them do
        if self.listing == "functions":
            d = self.results[self.subject]
            top = d.get("function", [])
            at = self.picked()
            if 0 <= at < len(top):
                term = top[at]["term"]
                self.chart.gather([n for n, _, _ in function.drivers(term, d["atlas"])],
                                  term)
            else:
                self.chart.focus(None)
            return
        names = self.selected_parcels()
        self.chart.focus(names[0] if names else None)

    def seek_parcel(self, index):
        # put the crosshair in the middle of it, which only mni can work out
        self.slices.lit = None
        index = self.picked(index)
        if index < 0:
            return
        if self.listing == "parcels":
            rows = self.parcel_rows
            if index >= len(rows):
                return
            wanted = rows[index]["parcel"]
        elif self.listing == "connectivity":
            if index >= len(self.at_risk):
                return
            wanted = self.at_risk[index]["parcel"]
        else:
            return
        if self.space != "mni":
            self.action_space()
        order = {p["name"]: i for i, p in enumerate(atlas.parcels())}
        self.slices.lit = order.get(wanted, -1) + 1
        if self.slices.center(wanted):
            self.draw()



    # ----------------------------------------------------------------- render

    def draw(self):
        if not self.scene:
            return
        if self.panel_width() != self._panel_width:
            self.refresh_inputs()
            self.refresh_layers()
        if self.query_one("#layers", Rows).size.width != self._layers_width:
            self.refresh_layers()
        listing = self.query_one("#parcels", ListView)
        if listing.size.width != self._list_width:
            self._list_width = listing.size.width
            index = listing.index
            self.refresh_list()
            listing.index = 0 if index is None else index
        cell = get_cell_size()
        viewer = self.query_one("#viewer")
        if self.mode == "3d":
            widget = self.query_one("#canvas", Canvas)
            size = widget.content_size
            w, h = max(size.width * cell.width, 64), max(size.height * cell.height, 64)
            self.scene.resize(w, h)
            widget.image = self.scene.image()
            viewer.border_title = f"3d · {self.palette} colors"
            viewer.styles.border_subtitle_align = "right"
            other = "yba" if self.palette == "contrast" else "contrast"
            viewer.border_subtitle = keyed(f"v slices · p {other} colors")
        elif self.mode == "graph":
            if self.chart is None:
                self.chart = render.Chart(self.subject,
                                          self.results[self.subject]["atlas"])
                self.light_node()
            widget = self.query_one("#board", Board)
            size = widget.content_size
            widget.image = self.chart.image(
                (max(size.width * cell.width, 64), max(size.height * cell.height, 64)))
            self.scene.show_tract(self.chart.tract)
            viewer.styles.border_subtitle_align = "right"
            viewer.border_title = "graph"
            other = "spread" if self.chart.plan == "circle" else "circle"
            viewer.border_subtitle = keyed(
                f"v 3d \u00b7 s {other} \u00b7 click the mass to reset")
        else:
            for which in render.Slices.PLANES:
                pane = self.query_one(f"#p-{which}", Pane)
                size = pane.content_size
                pane.image = self.slices.plane(
                    which, (max(size.width * cell.width, 32), max(size.height * cell.height, 32)))
            self.refresh_readout()
            r = self.slices.readout()
            who = "" if self.watching is None else f" · {self.watching[10:]}"
            colors = f" · {self.palette} colors" if self.slices.involved else ""
            viewer.border_title = f"slices · {r['space']}{who}{colors}"
            strip, taken = self.strip(
                "viewer", [(str(i), m) for i, m in enumerate(MODALITIES, start=1)],
                r["modality"], absent=[m for m in MODALITIES if m not in r["modalities"]])
            viewer.styles.border_subtitle_align = "left"
            viewer.border_subtitle = self.band(viewer, strip, taken, self.slice_hints())

    def refresh_readout(self):
        r = self.slices.readout()
        w = max(self.query_one("#readout", Static).size.width, 24)
        place = " ".join(str(v) for v in r["coords"]) if r["coords"] else "—"
        lines = [f"[{FAINT}]{row(r['coord_space'], place, w)}[/]",
                 f"[{FAINT}]{row('voxel', ' '.join(str(v) for v in r['voxel']), w)}[/]",
                 f"[{DIM}]{row('label', r['label'] if r['segmented'] else '—', w)}[/]"]
        for note, ready in (("not stripped yet", r["stripped"]),
                            ("not segmented yet", r["segmented"])):
            if not ready:
                lines.append(f"[{WARN}]{row('', note, w)}[/]")

        # every sequence at this voxel, which is the comparison you actually make
        lines += ["", f"[{FAINT}]signal[/]"]
        for name, value in r["signal"].items():
            style = FG if name == r["modality"] else FAINT
            lines.append(f"[{style}]  {name}[/][{DIM}]{value:>{w - len(name) - 2}}[/]")

        if r["parcel"]:
            info = self.parcel_info().get(r["parcel"], {})
            lines += ["", f"[{FAINT}]parcel[/]", f"[{FG}]  {r['parcel']}[/]",
                      f"[{DIM}]  {info.get('long_name', '')[:w - 2]}[/]",
                      f"[{FAINT}]  {info.get('lobe', '')}[/]"]
        elif self.space == "mni":
            lines += ["", f"[{FAINT}]outside every parcel[/]"]
        else:
            lines += ["", f"[{FAINT}]parcels need mni, press s[/]"]
        self.query_one("#readout", Static).update("\n".join(lines))

    @staticmethod
    def parcel_info():
        return {p["name"]: p for p in atlas.parcels()}

    def strip(self, which, entries, chosen, absent=()):
        # key bright when it is the one you are on, grayed when it is not offered
        out, x, spans = [], Bordered.MARGIN, []
        for key, name in entries:
            if out:
                out.append(" " * Bordered.SPACING)
                x += Bordered.SPACING
            here = name not in absent
            tone = FG if name == chosen else FAINT if here else GONE
            out.append(f"[{tone}]{key}[/] [{DIM if here else GONE}]{name}[/]")
            spans.append((x, x + len(key) + 1 + len(name), name))
            x += len(key) + 1 + len(name)
        self.strip_spans[which] = spans
        return "".join(out), x - Bordered.MARGIN

    def click_strip(self, which, x):
        for start, end, name in self.strip_spans.get(which, []):
            if start <= x < end:
                if which == "viewer":
                    self.pick_modality(name)
                else:
                    self.show_list(name)
                return

    def slice_hints(self):
        # each key names what it gives you, not what you are looking at
        core = f"v graph · s {'native' if self.space == 'mni' else 'mni'}"
        if self.slices.involved:
            other = "yba" if self.palette == "contrast" else "contrast"
            return [f"{core} · p {other} colors", core]
        return [core]

    def band(self, widget, left, taken, choices):
        # one border line, a label at each end, the line itself drawn between
        room = widget.outer_size.width - 6 - taken
        for text in choices:
            if len(text) + 5 <= room:
                return left + f" [{DIM}]{'─' * (room - len(text) - 2)}[/] {keyed(text)}"
        return left

    def pick_modality(self, name):
        if self.mode == "slices" and name in self.slices.readout()["modalities"]:
            self.slices.modality = name
            self.draw()

    def scroll_shelf(self, column, line, by):
        # the wheel scrolls the tract list when the pointer is over it, and
        # zooms the chart when it is not
        if self.mode != "graph" or self.chart is None:
            return False
        cell = get_cell_size()
        x, y = column * cell.width, line * cell.height
        if not self.chart.over_shelf(x, y):
            return False
        self.chart.scroll_shelf(by)
        self.draw()
        return True

    def pick_node(self, column, line):
        # a node clicked in the chart becomes the selected row in the list
        if self.mode != "graph" or self.chart is None:
            return
        cell = get_cell_size()
        x = column * cell.width + cell.width // 2
        y = line * cell.height + cell.height // 2
        # a row in the inventory is one tract, and picking it is picking that
        # tract everywhere, so the 3d can show the same line
        tract = self.chart.picked_row(x, y)
        if tract is not None:
            self.chart.tract = None if tract == self.chart.tract else tract
            self.draw()
            return
        at = self.chart.pick(x, y)
        if at is None:
            return
        if at >= len(self.chart.nodes):
            # neither hub is a parcel, so clicking either one clears
            self.chart.home()
            self.chart.focus(None)
            self.chart.tract = None
            self.scene.show_tract(None)
            self.draw()
            return
        name = self.chart.name(at)
        if name is None:
            return
        # a parcel the tumor overlaps is a row in the parcel list, one it
        # missed is a row in connectivity, so open whichever list holds it
        overlapped = self.chart.nodes[at] in self.chart.inside
        wanted = "parcels" if overlapped else "connectivity"
        where = self.rows_of(wanted)
        if name not in where:
            # too weakly tied for a connectivity row, so light it where it is
            self.chart.focus(name)
            self.draw()
        elif self.listing == wanted:
            self.select_row(where.index(name) + 1)
        else:
            self.action_show_list(wanted)
            self.call_after_refresh(self.select_row, where.index(name) + 1)

    def rows_of(self, listing):
        # what a list is offering, without having had to show it yet
        if listing == "parcels":
            return [r["parcel"] for r in self.shown_parcels()]
        return [c["parcel"] for c in connect.at_risk(self.results[self.subject]["atlas"])]

    def pick_parcel(self, column, line):
        # a parcel clicked in the 3d becomes the selected row in the list
        if self.mode != "3d" or not self.scene:
            return
        cell = get_cell_size()
        name = self.scene.pick(column * cell.width + cell.width // 2,
                               line * cell.height + cell.height // 2)
        if name is None:
            return
        names = [r["parcel"] for r in self.parcel_rows]
        if name not in names:
            return
        if self.listing != "parcels":
            self.listing = "parcels"
            self.refresh_list()
            self.call_after_refresh(self.select_row, names.index(name) + 1)
        else:
            self.select_row(names.index(name) + 1)

    def select_row(self, index):
        self.query_one("#parcels", ListView).index = index

    def queue_chart(self, zoom=1.0, pan=(0.0, 0.0)):
        # the terminal sends a move event per pixel, so collect and draw on a clock
        if self.mode != "graph" or self.chart is None:
            return
        self.waiting[0] *= zoom
        self.waiting[1] += pan[0]
        self.waiting[2] += pan[1]
        if self._spin is None:
            self._spin = self.set_interval(1 / 30, self.flush_chart)
        if self._settle is not None:
            self._settle.stop()
        self._settle = self.set_timer(0.3, self.settle_chart)

    def flush_chart(self):
        zoom, across, down = self.waiting
        self.waiting = [1.0, 0.0, 0.0]
        if zoom == 1.0 and not across and not down:
            return
        self.chart.move(zoom, (across, down))
        self.draw()

    def settle_chart(self):
        if self._spin is None:
            return
        self._spin.stop()
        self._spin = None
        self.flush_chart()

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

    VIEW_ACTIONS = ("view", "space", "palette", "show_list")

    def check_action(self, action, parameters):
        # gray a key out rather than let it look like it did nothing
        if action in self.VIEW_ACTIONS and not self.scene:
            return None
        if action == "palette":
            # the slices only carry parcels in mni, where the colors mean something
            return (self.mode == "3d" or (self.mode == "slices" and self.space == "mni")) or None
        if action == "space":
            return self.mode in ("slices", "graph") or None
        if action == "cancel":
            return self.proc is not None or None
        if action == "show_list":
            return bool(self.scene)
        return True

    def action_view(self):
        self.mode = VIEWERS[(VIEWERS.index(self.mode) + 1) % len(VIEWERS)]
        self.query_one("#canvas").display = self.mode == "3d"
        self.query_one("#grid").display = self.mode == "slices"
        self.query_one("#board").display = self.mode == "graph"
        # a tract picked in the chart stays picked in the 3d and the other way
        if self.scene and self.chart is not None:
            self.scene.show_tract(self.chart.tract)
        if self.scene:
            self.focus_row()
        self.refresh_bindings()
        self.call_after_refresh(self.draw)

    def action_space(self):
        # the graph has two arrangements of the same data, the way the slices
        # have two spaces, so the same key swaps whichever one is showing
        if self.mode == "graph":
            if self.chart is not None:
                self.chart.swap()
                self.draw()
            return
        self.space = "mni" if self.space == "native" else "native"
        self.slices.load(self.space)
        self.sync_overlay()
        self.refresh_bindings()
        if self.mode == "slices":
            self.draw()

    def action_palette(self):
        self.palette = "yba" if self.palette == "contrast" else "contrast"
        self.scene.recolor(self.palette)
        self.sync_overlay()
        self.refresh_list()
        self.draw()

    def action_show_list(self, which):
        if which == self.listing:
            return
        self.listing = which
        if self.scene:
            self.scene.spotlight(None)
        self.refresh_list()
        self.draw()

    def show_list(self, which):
        self.action_show_list(which)

    def save_view(self):
        # whatever the viewer is showing, at a size worth sending to someone
        if not self.scene:
            self.log_write(f"[{BAD}]nothing loaded to save[/]")
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if self.mode == "3d":
            image, kind = self.scene.snapshot(), "3d"
        else:
            image, kind = self.slices.montage(), f"slices-{self.space}"
        dest = out_dir(self.subject) / f"{self.subject}_{kind}_{stamp}.png"
        image.save(dest)
        self.log_write(f"[{OK}]wrote {dest.name}[/] [{FAINT}]{image.size[0]}x{image.size[1]}[/]")

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
        stage, timepoint = name if name in STAGES else "fetch", 1
        for flag in ("--from", "--timepoint"):
            if flag in args:
                cut = args.index(flag)
                value = args[cut + 1] if cut + 1 < len(args) else ""
                args = args[:cut] + args[cut + 2:]
                if flag == "--from":
                    stage = value
                elif value in ("1", "2"):
                    timepoint = int(value)
                else:
                    self.log_write(f"[{BAD}]timepoint is 1 or 2[/]")
                    return
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
                self.run_pipeline(numbers, stage, timepoint)
        elif name in ("show", "load"):
            self.show_subject(numbers, timepoint)
        elif name in ("save", "export"):
            self.save_view()
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

    def show_subject(self, numbers, timepoint=1):
        if not numbers:
            self.log_write(f"[{BAD}]usage: show <subject number>[/]")
            return
        wanted = subject_id(numbers[0], timepoint)
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
        if not self.proc:
            self.log_write(f"[{FAINT}]nothing running[/]")
            return
        self.proc.terminate()
        # the segmenter runs detached, so it needs stopping in its own right
        stopped = segment.stop()
        self.log_write(f"[{WARN}]canceled{', stopped the segmenter' if stopped else ''}[/]")

    def busy(self, note):
        self.query_one("#shell").border_subtitle = keyed(note)

    def tick(self):
        done = STAGES.index(self.stage) + 1 if self.stage in STAGES else 0
        where = f"{self.stage} {bar(done / len(STAGES), 5)} {done}/{len(STAGES)}"
        self.busy(f"{where} · {spell(round(time.time() - self.started))} · x cancels")

    def preview(self, subject):
        # the scans exist from here on, so look at the case instead of a stale one
        if not stage_file(subject, "fetch").exists() or subject == self.subject:
            return
        self.slices = render.Slices(subject, self.space)
        self.watching = subject
        if self.mode == "3d":
            self.action_view()
        else:
            self.draw()

    @work(exclusive=True)
    async def run_pipeline(self, numbers, stage="fetch", timepoint=1):
        self.started = time.time()
        self._clock = self.set_interval(1.0, self.tick)
        self.tick()
        # --no-view because this panel already has the viewer next to it
        call = ["main.py", *[str(n) for n in numbers], "--no-view"]
        if stage != "fetch":
            call += ["--from", stage]
        if timepoint != 1:
            call += ["--timepoint", str(timepoint)]
        self.log_write(f"[{FAINT}]{' '.join(call)}[/]")
        # rich draws a live spinner that becomes escape-code debris in a log widget
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, *call, cwd=str(ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        self.refresh_bindings()
        async for raw in self.proc.stdout:
            text = ANSI.sub("", raw.decode(errors="replace")).rstrip()
            if not text:
                continue
            found = MARKER.match(text)
            if found:
                self.stage = found.group(1)
                self.tick()
            elif SUBJECT.match(text):
                self.log_write(f"[{FG}]{text}[/]")
                self.preview(text)
            else:
                self.log_write(f"[{DIM}]{text}[/]")
        code = await self.proc.wait()
        elapsed = time.time() - self.started
        self.proc, self.stage = None, None
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
        self.load_subject(subject_id(numbers[-1], timepoint))


if __name__ == "__main__":
    Console().run()
