#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Terminal UI for the segmentation pipeline. Needs kitty for the graphics protocol.

import csv
import json
import os
import time
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, VerticalScroll
from textual.widgets import Footer, Static
from textual_image._terminal import get_cell_size
from textual_image.renderable import Image as Renderable
from textual_image.widget import Image as ImageWidget

from analysis import measure
from show import compare
from show import render
from paths import (ATLAS_INPUT, LABELS, MODALITIES, REGIONS, RESULTS, results_dir,
                   STAGES, out_dir, raw_dir, stage_file, stripped_dir)


# gruvbox dark, the same values btop reads out of gruvbox_dark_v2.theme
FG, DIM, FAINT = "#ebdbb2", "#a89984", "#7c6f64"
WARN, BAD, GONE = "#fabd2f", "#fb4934", "#504945"
SHELL_OPACITY = 0.15

# the renderer's colors as hex for the text panels; 0 is the whole tumor
TISSUE = {LABELS.get(k, "WT"): "#%02x%02x%02x" % rgb for k, rgb in render.TISSUE.items()}
# each scan's tumor as one shape, with its parts under it
SCAN_PARTS = (("WT", "tumor", 0), ("SNFH", "FLAIR hyperintense", 1), ("ET", "enhancing", 1),
              ("NETC", "non-enhancing core", 1), ("RC", "cavity", 1))
SCANS = {"1": "scan 1 \u00b7 pre-op", "2": "scan 2 \u00b7 follow-up"}


def scan_rows(scan):
    return [(f"{scan}:{name}", label, TISSUE[name], deep) for name, label, deep in SCAN_PARTS]


# a key starting -- is a heading. Plans are always built from scan 1, the scan a
# plan is drawn on, whichever scan is in the background
LAYERS = [("shell", "brain surface", "#a89984", 0),
          ("--1", SCANS["1"], "", 0),
          *scan_rows("1"),
          ("--p", "plans from scan 1", "", 0),
          ("T:46", "NRG 2 cm (46 Gy)", "#%02x%02x%02x" % render.NRG["46"], 0),
          ("T:60", "NRG 2 cm (60 Gy)", "#%02x%02x%02x" % render.NRG["60"], 0),
          ("W:46", "walk-down 46 Gy", "#%02x%02x%02x" % render.WALK["46"], 0),
          ("W:60", "walk-down 60 Gy", "#%02x%02x%02x" % render.WALK["60"], 0),
          ("W:ends46", "walk endpoints 46 Gy", "#%02x%02x%02x" % render.WALK["46"], 1),
          ("W:ends60", "walk endpoints 60 Gy", "#%02x%02x%02x" % render.WALK["60"], 1),
          ("--2", SCANS["2"], "", 0),
          *scan_rows("2"),
          ("C:both", "in both scans", "#ebdbb2", 0),
          ("C:new", "new at follow-up", "#fb4934", 0),
          ("C:gone", "gone at follow-up", "#8ec07c", 0),
          ("--t", "tracts through the tumor shown", "", 0),
          ("tracts:1", "through pre-op tumor", "#%02x%02x%02x" % render.TRACTS["1"], 0),
          ("tracts:2", "through follow-up tumor", "#%02x%02x%02x" % render.TRACTS["2"], 0)]
KEYS = {k for k, *_ in LAYERS if not k.startswith("--")}
# on the first ever start; after that the viewer remembers
SHOWN = {"shell", "1:WT", "2:WT", "tracts:1", "tracts:2"}
# what the viewer remembers between patients and between runs; a personal
# setting, so it lives with the user's config rather than with the results
STATE = Path.home() / ".config" / "brats-tui" / "state.json"
# WT, TC and ET are unions of the tissue classes, so they are deliberately left
# uncolored: any class color on them reads as "this row is that tissue"
ATLAS_FILES = [("YBA_696.nii", "volume", "696"),
               ("hcp1065/trk", "tracts", "HCP-1065")]
VIEWERS = ("3d", "slices")
def target_ml(subject, phase, target="walk"):
    # a target's patient-space volume, as analysis/walkdown.py scored it: the
    # walk-down ("walk") or the NRG 2 cm volume it is compared with ("iso")
    path = results_dir("native") / "walkdown.csv"
    if not path.exists():
        return ""
    stamp = path.stat().st_mtime
    if _walked.get("stamp") != stamp:
        with open(path) as fh:
            _walked.clear()
            _walked.update({(r["patient"], r["phase"], r["target"]): r["volume_ml"]
                            for r in csv.DictReader(fh)})
        _walked["stamp"] = stamp
    value = _walked.get((subject[10:15], phase, target))
    return f"{value} mL" if value else ""


_walked = {}


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


class Bordered(VerticalScroll):
    # A strip is drawn into the bottom border, so its clicks land here.
    MARGIN, SPACING = 3, 2

    def on_click(self, event):
        # clicks arrive in outer coordinates, so the bottom border is the last row
        if event.y == self.outer_size.height - 1:
            self.app.click_strip(self.id, event.x)


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
    .pane { height: 1fr; border: round #504945; border-title-color: #a89984; }
    #readout { height: 1fr; border: round #504945; border-title-color: #a89984;
               padding: 0 1; }
    #layers { height: auto; padding: 0 1; }
    #inputs {
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
        ("n", "background", "scan mri"),
    ]

    def __init__(self):
        super().__init__()
        self.results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        self.subjects = sorted(self.results)
        self.subject = self.subjects[0] if self.subjects else None
        self.mode, self.space = "3d", "native"
        self.scene = self.slices = None
        self.later, self.background = None, "1"
        self._timer = None
        self.hidden_layers, self.subject_rows = KEYS - SHOWN, {}
        self.restore_state()
        self.layer_rows = list(LAYERS)
        self.pending, self._spin, self._settle = [0.0, 0.0, 1.0], None, None
        self.strip_spans = {}
        self._panel_width = self._layers_width = 0

    # ---------------------------------------------------------------- compose

    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            with VerticalScroll(id="inputs", classes="box"):
                yield Rows(self.click_subject, id="stats")
            with Bordered(id="viewer", classes="box"):
                yield Canvas(id="canvas")
                with Grid(id="grid"):
                    yield Pane("axial", id="p-axial", classes="pane")
                    yield Pane("coronal", id="p-coronal", classes="pane")
                    yield Pane("sagittal", id="p-sagittal", classes="pane")
                    yield Static("", id="readout")
            with Bordered(id="right", classes="box"):
                yield Rows(self.click_layer, id="layers")
        yield Footer()

    def on_mount(self):
        self.query_one("#grid").display = False
        self.query_one("#inputs").border_title = "inputs"
        self.query_one("#inputs").border_subtitle = "click a patient"
        self.query_one("#readout").border_title = "cursor"
        self.query_one("#right").border_title = "layers"
        if not os.environ.get("TERM", "").startswith("xterm-kitty"):
            self.say("not kitty: images will render as text blocks", "warning")
        if self.subject:
            self.load_subject(self.subject)
            self.show_mode()
            self.call_after_refresh(self.draw)
        else:
            self.say("nothing segmented yet, run ./main.py 36 in a terminal", "warning")

    def say(self, text, severity="information"):
        # the pipeline runs in a terminal of its own now, so only a few things
        # are ever worth saying here, and a toast is enough for them
        self.notify(text, severity=severity)

    def schedule_draw(self):
        # resizing fires a burst of events; only redraw once it settles
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_timer(0.06, self.draw)

    # ---------------------------------------------------------------- subject

    def scans_of(self, subject):
        # the patient's pre-op scan, and the follow-up when there is one
        group = [s for s in self.subjects if s[10:15] == subject[10:15]]
        return group[0], (group[1] if len(group) > 1 else None)

    def load_subject(self, subject):
        # any scan of a patient loads the patient: both scans, side by side
        first, later = self.scans_of(subject)
        # results.json can outlive the files it describes, and that is not fatal
        for who in (first, later):
            missing = [name for name in ("seg.nii.gz", "seg_in_mni.nii.gz")
                       if who and not (out_dir(who) / name).exists()]
            if missing:
                self.say(f"{who}: {', '.join(missing)} is gone, "
                         f"rerun with ./main.py {int(who[10:15])}", "error")
                return False
        self.subject, self.later = first, later
        self.scene = render.Scene(first, later=later)
        self.scene.set_opacity("shell", SHELL_OPACITY)
        self.slices = render.Slices(first, self.space, later=later, background=self.background)
        # the layers kept on from the last patient are built for this one too
        self.ensure_layers(KEYS - self.hidden_layers)
        self.apply_layers()
        self.refresh_inputs()
        self.refresh_layers()
        self.draw()
        self.save_state()
        return True

    def ensure_layers(self, keys):
        # the layers that are only built when first asked for
        for kind, prefix in render.TARGET_KEY.items():
            for phase in render.WALK_PHASES:
                if prefix + phase in keys:
                    self.scene.ensure_target(kind, phase)
        for phase in render.WALK_PHASES:
            if "W:ends" + phase in keys:
                self.scene.ensure_ends(phase)
        if self.later and any(k.startswith("C:") for k in keys):
            self.scene.ensure_compare(self.subject, self.later)

    def restore_state(self):
        # the last session's layers and views, so nothing resets between runs
        try:
            saved = json.loads(STATE.read_text())
        except (OSError, ValueError):
            return
        shown = set(saved.get("shown", []))
        # the tracts were one layer before they were split by scan
        if "tracts" in shown:
            shown |= {"tracts:1", "tracts:2"}
        # and the walk endpoints were one layer before they were split by phase
        if "W:ends" in shown:
            shown |= {"W:ends46", "W:ends60"}
        # and the NRG volume was one layer, the 46 Gy one, before the boost joined it
        if "T:margin" in shown:
            shown.add("T:46")
        shown &= KEYS
        self.hidden_layers = KEYS - shown
        self.mode = saved.get("mode", self.mode) if saved.get("mode") in VIEWERS else self.mode
        self.space = saved.get("space", self.space) if saved.get("space") in ("native", "mni") else self.space
        self.background = saved.get("background", self.background)
        if saved.get("subject") in self.results:
            self.subject = saved["subject"]

    def save_state(self):
        try:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps({
                "shown": sorted(KEYS - self.hidden_layers), "mode": self.mode,
                "space": self.space,
                "background": self.background, "subject": self.subject}, indent=1))
        except OSError:
            pass

    def panel_width(self):
        # The panel loses a column whenever its scrollbar appears.
        width = self.query_one("#stats", Rows).size.width
        return width if width > 12 else 30

    def refresh_inputs(self):
        w = self._panel_width = self.panel_width()
        out = [f"[{FG}]{row('patient  days', 'whole tumor', w)}[/]"]
        self.subject_rows = {}
        patients = {}
        for name in self.subjects:
            patients.setdefault(name[10:15], []).append(name)
        for n, (patient, scans) in enumerate(sorted(patients.items()), start=1):
            here = patient == self.subject[10:15]
            self.subject_rows[n] = scans[0]
            sizes = [self.results[s]["regions_ml"]["WT"] for s in scans]
            right = " \u2192 ".join(f"{v:.0f}" for v in sizes) + " mL"
            # days between the scans, from the release's own table
            days = measure.interval(scans[0]) if len(scans) > 1 else None
            left = ("\u203a " if here else "  ") + patient + (f" {days:>4}d" if days else "")
            out.append(f"[{FG if here else DIM}]{row(left, right, w)}[/]")

        def rule(title):
            return f"[{FAINT}]\u2500\u2500 [/][{FG}]{title}[/][{FAINT}] {'\u2500' * (w - 4 - len(title))}[/]"

        scans = [(tag, who) for tag, who in (("1", self.subject), ("2", self.later)) if who]
        out.append(rule("stats"))
        for tag, who in scans:
            d = self.results[who]
            r = d["regions_ml"]
            out.append(f"[{DIM}]{SCANS[tag]}[/]")
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
        if self.later:
            before, after = self.subject, self.later
            out.append(f"[{DIM}]changes[/]")
            change = (self.results[after]["regions_ml"]["WT"]
                      - self.results[before]["regions_ml"]["WT"])
            out.append(f"[{WARN if change > 0 else FAINT}]"
                       f"{row('  scan 1 \u2192 2 WT', f'{change:+.1f} mL', w)}[/]")
            same = compare.against(before, after)
            out.append(f"[{FAINT}]{row('  dice', f"{same['dice']:.2f}", w)}[/]")
            out.append(f"[{FAINT}]{row('  of scan 2 inside scan 1', f"{same['inside']:.0%}", w)}[/]")

        out.append(f"[{DIM}]run[/]")
        for tag, who in scans:
            d = self.results[who]
            run = d.get("run", {})
            made = f"  {tag}: {run.get('algorithm', 'not recorded')}"
            out.append(f"[{FAINT}]{row(made, spell(run['seconds']) if 'seconds' in run else '', w)}[/]")
            if "kept" in run:
                out.append(f"[{FAINT}]{row('     aligned on', f"{run['kept']:.0%}", w)}[/]")

        out.append(rule("checks"))
        for tag, who in scans:
            d = self.results[who]
            out.append(f"[{DIM}]{SCANS[tag]}[/]")
            for name, note, gap, trouble in self.stage_checks(who, d, d.get("run", {})):
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

    def stage_checks(self, subject, d, run):
        # name, what it says, and what is wrong if anything
        times, stamps, out = run.get("times", {}), {}, []
        for stage in STAGES:
            if stage == "align" and subject.endswith("_11"):
                continue
            path = stage_file(subject, stage)
            stamps[stage] = (run.get("mapped") if path is None
                             else path.stat().st_mtime if path.exists() else None)
            trouble = self.gap(subject, stage, d)
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

    def gap(self, subject, stage, d):
        # the specific thing missing, which is what the red sub row says
        if stage == "fetch":
            short = [m for m in MODALITIES
                     if not (raw_dir(subject) / f"{subject}_{m}.nii.gz").exists()]
            return ", ".join(short) + " not downloaded" if short else None
        if stage == "strip":
            short = [m for m in MODALITIES
                     if not (stripped_dir(subject) / f"{subject}_{m}.nii.gz").exists()]
            if short:
                return ", ".join(short) + " not stripped"
            found = out_dir(subject) / "verify.txt"
            if found.exists() and found.read_text().strip():
                return found.read_text().strip().splitlines()[0]
            return None
        if stage == "map" and "atlas" not in d:
            return "no parcel overlap"
        return None

    def click_subject(self, row_index):
        chosen = self.subject_rows.get(row_index)
        if chosen and chosen != self.subject:
            self.load_subject(chosen)

    # ----------------------------------------------------------------- layers

    def scan_subject(self, scan):
        return self.subject if scan == "1" else self.later

    def refresh_layers(self):
        out = []
        self._layers_width = self.query_one("#layers", Rows).size.width
        w = max(self._layers_width, 30) - 4
        walked = render.target_file(self.subject, "walk", "46", "mni") is not None
        scored = render.target_file(self.subject, "nrg", "46", "mni") is not None

        def offered(key):
            if key in ("--2", "C:both", "C:new", "C:gone", "tracts:2") or key.startswith("2:"):
                if not self.later:
                    return False
            if key.startswith("W:"):
                return walked
            if key.startswith("T:"):
                return scored
            if key[:2] in ("1:", "2:") and not key.endswith(":WT"):
                return key[2:] in self.results[self.scan_subject(key[0])]["labels_ml"]
            return True

        self.layer_rows = [r for r in LAYERS if offered(r[0])]
        for key, name, color, deep in self.layer_rows:
            if key.startswith("--"):
                title = f" {name} "
                out.append(f"[{FAINT}]\u2500\u2500[/][{DIM}]{title}[/]"
                           f"[{FAINT}]{'\u2500' * max(w + 2 - len(title), 0)}[/]")
                continue
            on = key not in self.hidden_layers
            if key.startswith("tracts:"):
                count = self.scene.tract_count.get(key[7:]) if self.scene else None
                value = f"{count:,}" if count is not None and on else ""
            elif key.startswith("C:"):
                value = f"{compare.against(self.subject, self.later)[key[2:]]} mL"
            elif key.startswith("T:"):
                value = target_ml(self.subject, key[2:], "iso")
            elif key.startswith("W:ends"):
                value = f"{len(render.walk_ends(self.subject).get(key[6:], ())):,}"
            elif key.startswith("W:"):
                value = target_ml(self.subject, key[2:])
            elif key.endswith(":WT"):
                value = f"{self.results[self.scan_subject(key[0])]['regions_ml']['WT']} mL"
            elif key[:2] in ("1:", "2:"):
                value = self.results[self.scan_subject(key[0])]["labels_ml"].get(key[2:])
                value = f"{value} mL" if value else ""
            else:
                value = ""
            out.append("  " * deep + f"[{color if on else FAINT}]"
                       f"{'[\u00d7]' if on else '[ ]'}[/] "
                       f"[{DIM if on else FAINT}]{row(name, value, w - 2 * deep)}[/]")
        self.query_one("#layers", Rows).update("\n".join(out))

    def apply_layers(self):
        # the one place that turns the checkbox states into what is drawn
        if not self.scene:
            return
        tracts = {k for k, *_ in LAYERS if k.startswith("tracts:")}
        self.scene.show_crossing({k[7:] for k in tracts - self.hidden_layers})
        for key, *_ in LAYERS:
            if key.startswith("--"):
                continue
            on = key not in self.hidden_layers
            # the combined shape stands in for the parts when none of them is on
            if key.endswith(":WT"):
                on = on and not self.split(key[0])
            if key in self.scene.actors:
                self.scene.set_visible(key, on)
        self.sync_overlay()

    def split(self, scan):
        # that scan's parts that are on and have something to draw
        return [f"{scan}:{name}" for name, _, deep in SCAN_PARTS
                if deep and f"{scan}:{name}" not in self.hidden_layers
                and f"{scan}:{name}" in self.scene.actors]

    def click_layer(self, row_index):
        if not (0 <= row_index < len(self.layer_rows) and self.scene):
            return
        key = self.layer_rows[row_index][0]
        if key.startswith("--"):
            return
        scan = key[0] if key[:2] in ("1:", "2:") else None
        tumor = {f"{t}:{name}" for t in ("1", "2") for name, *_ in SCAN_PARTS}
        change = {k for k, *_ in LAYERS if k.startswith("C:")}
        if scan and key.endswith(":WT"):
            # the parent carries its parts with it
            family = {f"{scan}:{name}" for name, *_ in SCAN_PARTS}
            self.hidden_layers = (self.hidden_layers - family if key in self.hidden_layers
                                  else self.hidden_layers | family)
        else:
            self.hidden_layers ^= {key}
            if scan and key not in self.hidden_layers:
                self.hidden_layers.discard(f"{scan}:WT")
        # the comparison redraws the same tissue, so it and the tumors take turns.
        # nothing else is a rival: plans, tracts and the shell sit over any of it
        if key not in self.hidden_layers:
            if key in tumor:
                self.hidden_layers |= change
            elif key in change:
                self.hidden_layers |= tumor
            self.ensure_layers({key})
        self.apply_layers()
        self.refresh_layers()
        self.draw()
        self.save_state()

    def sync_overlay(self):
        # the same checkbox hides a class in the slices as in the 3d
        if not self.slices:
            return
        for scan in ("1", "2"):
            self.slices.hidden[scan] = {value for value, name in LABELS.items()
                                        if f"{scan}:{name}" in self.hidden_layers}
            whole = f"{scan}:WT" not in self.hidden_layers and not (self.scene and self.split(scan))
            self.slices.whole[scan] = render.TISSUE[0] if whole else None
        self.slices.show_targets = {(kind, p) for kind, prefix in render.TARGET_KEY.items()
                                    for p in render.WALK_PHASES if prefix + p not in self.hidden_layers}
        self.slices.compare = {}
        wanted = [k[2:] for k, *_ in LAYERS
                  if k.startswith("C:") and k not in self.hidden_layers]
        if self.later and wanted:
            # in mni the pieces are already on the grid; in native they go onto the follow-up's
            source = (compare.masks(self.subject, self.later) if self.space == "mni"
                      else compare.native(self.later, self.subject, self.later))
            for name in wanted:
                if name in source:
                    self.slices.compare[name] = self.slices.canon_mask(source[name])

    # ----------------------------------------------------------------- render

    def draw(self):
        if not self.scene:
            return
        if self.panel_width() != self._panel_width:
            self.refresh_inputs()
            self.refresh_layers()
        if self.query_one("#layers", Rows).size.width != self._layers_width:
            self.refresh_layers()
        cell = get_cell_size()
        viewer = self.query_one("#viewer")
        if self.mode == "3d":
            widget = self.query_one("#canvas", Canvas)
            size = widget.content_size
            w, h = max(size.width * cell.width, 64), max(size.height * cell.height, 64)
            self.scene.resize(w, h)
            widget.image = self.scene.image()
            viewer.border_title = "3d"
            viewer.styles.border_subtitle_align = "right"
            viewer.border_subtitle = keyed("v slices")
        else:
            for which in render.Slices.PLANES:
                pane = self.query_one(f"#p-{which}", Pane)
                size = pane.content_size
                pane.image = self.slices.plane(
                    which, (max(size.width * cell.width, 32), max(size.height * cell.height, 32)))
            self.refresh_readout()
            r = self.slices.readout()
            mri = f" · scan {r['background']} mri" if self.later else ""
            viewer.border_title = f"slices · {r['space']}{mri}"
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
                 f"[{DIM}]{row('scan 1', r['label'] if r['segmented'] else '—', w)}[/]"]
        if r["label2"] is not None:
            lines.append(f"[{DIM}]{row('scan 2', r['label2'], w)}[/]")
        for note, ready in (("not stripped yet", r["stripped"]),
                            ("not segmented yet", r["segmented"])):
            if not ready:
                lines.append(f"[{WARN}]{row('', note, w)}[/]")

        # every sequence at this voxel, which is the comparison you actually make
        lines += ["", f"[{FAINT}]signal[/]"]
        for name, value in r["signal"].items():
            style = FG if name == r["modality"] else FAINT
            lines.append(f"[{style}]  {name}[/][{DIM}]{value:>{w - len(name) - 2}}[/]")

        self.query_one("#readout", Static).update("\n".join(lines))

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
                return

    def slice_hints(self):
        # each key names what it gives you, not what you are looking at
        core = f"v 3d · s {'native' if self.space == 'mni' else 'mni'}"
        if self.later:
            core += f" · n scan {'1' if self.slices.background == '2' else '2'} mri"
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

    VIEW_ACTIONS = ("view", "space", "background")

    def check_action(self, action, parameters):
        # gray a key out rather than let it look like it did nothing
        if action in self.VIEW_ACTIONS and not self.scene:
            return None
        if action == "space":
            return self.mode == "slices" or None
        if action == "background":
            return (self.mode == "slices" and self.later is not None) or None
        return True

    def show_mode(self):
        self.query_one("#canvas").display = self.mode == "3d"
        self.query_one("#grid").display = self.mode == "slices"

    def action_view(self):
        self.mode = VIEWERS[(VIEWERS.index(self.mode) + 1) % len(VIEWERS)]
        self.show_mode()
        self.save_state()
        self.refresh_bindings()
        self.call_after_refresh(self.draw)

    def action_space(self):
        self.space = "mni" if self.space == "native" else "native"
        self.slices.load(self.space)
        self.sync_overlay()
        self.refresh_bindings()
        self.save_state()
        if self.mode == "slices":
            self.draw()

    def action_background(self):
        # which scan's MRI the slices are drawn on; the tumors stay where they are
        self.background = "2" if self.background == "1" else "1"
        self.slices.background = self.background
        self.slices.load(self.space)
        self.sync_overlay()
        self.save_state()
        self.draw()

    def on_key(self, event: events.Key):
        if self.mode == "slices" and self.slices:
            if event.key in "1234":
                names = self.slices.readout()["modalities"]
                i = int(event.key) - 1
                if i < len(names):
                    self.slices.modality = names[i]
                    self.draw()


if __name__ == "__main__":
    Console().run()
