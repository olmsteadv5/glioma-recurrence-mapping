#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# Terminal UI for the segmentation pipeline. Needs kitty for the graphics protocol.

import asyncio
import json
import os
import re
import sys

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Label, ListItem, ListView, RichLog, Static

import function
import render
from paths import RESULTS, ROOT, subject_id

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")
BARS = 10


def bar(fraction, n=BARS):
    filled = round(max(0.0, min(1.0, fraction)) * n)
    return "█" * filled + "░" * (n - filled)


class Viewer(Static):
    # Holds whichever image is current; the app swaps what it renders.
    pass


class Console(App):
    CSS = """
    Screen { layout: horizontal; background: #141414; }
    #left { width: 26; border: round #31312f; }
    #mid { width: 1fr; }
    #right { width: 42; border: round #31312f; }
    #viewer { height: 1fr; border: round #31312f; }
    #image { height: 1fr; }
    #caption { height: 1; }
    #log { height: 14; border: round #31312f; }
    .title { color: #bfa87f; text-style: bold; }
    .dim { color: #8d8b85; }
    ListView { background: #141414; }
    ListItem { background: #141414; padding: 0 1; }
    ListItem.--highlight { background: #262420; }
    RichLog { background: #141414; }
    """
    BINDINGS = [
        ("q", "quit", "quit"),
        ("v", "toggle_view", "3d/slices"),
        ("s", "toggle_space", "native/mni"),
        ("p", "toggle_palette", "palette"),
        ("f", "toggle_list", "parcels/functions"),
        ("r", "run", "run pipeline"),
        ("c", "cancel", "cancel"),
    ]

    def __init__(self):
        super().__init__()
        self.results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        self.subject = sorted(self.results)[0] if self.results else None
        self.mode, self.space, self.palette, self.listing = "3d", "native", "contrast", "parcels"
        self.scene = self.slices = self.proc = None

    def compose(self) -> ComposeResult:
        with Vertical(id="left"):
            yield Label(" subjects", classes="title")
            yield ListView(*[ListItem(Label(s.replace("UPENN-GBM-", "")))
                             for s in sorted(self.results)], id="subjects")
            yield Static("", id="meta", classes="dim")
        with Vertical(id="mid"):
            with Vertical(id="viewer"):
                yield Viewer(id="image")
                yield Static("", id="caption", classes="dim")
            yield RichLog(id="log", wrap=True, markup=True, highlight=False)
        with Vertical(id="right"):
            yield Static("", id="breakdown")
        yield Footer()

    def on_mount(self):
        self.log_write(f"[#bfa87f]{len(self.results)} subjects[/] from {RESULTS}")
        if not os.environ.get("TERM", "").startswith("xterm-kitty"):
            self.log_write("[#f2c14e]not running in kitty; images will look poor[/]")
        if self.subject:
            self.load_subject(self.subject)

    def log_write(self, text):
        self.query_one("#log", RichLog).write(text)

    # ---------------------------------------------------------------- subject

    def load_subject(self, subject):
        self.subject = subject
        overlap = self.results[subject]["atlas"]
        self.log_write(f"loading {subject}")
        self.scene = render.Scene(subject, overlap, palette=self.palette)
        self.slices = render.Slices(subject, self.space)
        self.refresh_meta()
        self.refresh_breakdown()
        self.draw()

    def refresh_meta(self):
        d = self.results[self.subject]
        r, a = d["regions_ml"], d["atlas"]
        lines = [f"WT {r['WT']} mL", f"TC {r['TC']} mL", f"ET {r['ET']} mL", "",
                 f"{a['in_parcels_ml']} of {a['tumour_ml']} mL", f"in {len(a['parcels'])} parcels"]
        if d.get("dice"):
            lines += ["", "dice " + " ".join(f"{v}" for v in d["dice"].values())]
        self.query_one("#meta", Static).update("\n".join(" " + l for l in lines))

    def refresh_breakdown(self):
        d = self.results[self.subject]
        out = []
        if self.listing == "parcels":
            out.append("[#bfa87f] parcels involved[/]\n")
            for p in d["atlas"]["parcels"][:26]:
                out.append(f" {p['parcel']:<11}{bar(p['fraction'])} {p['fraction']:>4.0%}")
        else:
            out.append("[#bfa87f] functions at risk[/]")
            out.append("[#5b5a56] parcelsynth, meta-analytic[/]\n")
            top = d.get("function", [])
            peak = max((f["enrichment"] for f in top), default=1)
            for f in top[:20]:
                out.append(f" {f['term']:<16}{bar(f['enrichment']/peak)} x{f['enrichment']:<5.1f}")
        self.query_one("#breakdown", Static).update("\n".join(out))

    @on(ListView.Highlighted, "#subjects")
    def pick(self, event):
        if event.item is None:
            return
        chosen = sorted(self.results)[event.list_view.index or 0]
        if chosen != self.subject:
            self.load_subject(chosen)

    # ----------------------------------------------------------------- render

    def draw(self):
        widget = self.query_one("#image", Viewer)
        cell = getattr(self.app, "_cell_size", None) or (8, 16)
        w = max(widget.size.width * cell[0], 64)
        h = max(widget.size.height * cell[1], 64)
        if self.mode == "3d":
            self.scene.resize(w, h)
            image = self.scene.image()
            caption = f" 3d atlas  ·  {self.palette}"
        else:
            image = self.slices.plane("axial", (w, h))
            r = self.slices.readout()
            coords = f"{r['coords']}" if r["coords"] else r["coord_space"]
            caption = (f" {r['space']}  ·  {r['modality']}  ·  {r['label']}  ·  {coords}")
        widget.update(self._render_image(image))
        self.query_one("#caption", Static).update(caption)

    def _render_image(self, image):
        from textual_image.renderable import Image as Renderable
        return Renderable(image)

    def on_resize(self, event: events.Resize):
        if self.scene:
            self.draw()

    # ---------------------------------------------------------------- actions

    def action_toggle_view(self):
        self.mode = "slices" if self.mode == "3d" else "3d"
        self.draw()

    def action_toggle_space(self):
        self.space = "mni" if self.space == "native" else "native"
        self.slices.load(self.space)
        self.log_write(f"slices now {self.space}")
        if self.mode == "slices":
            self.draw()

    def action_toggle_palette(self):
        self.palette = "yba" if self.palette == "contrast" else "contrast"
        self.scene = render.Scene(self.subject, self.results[self.subject]["atlas"],
                                  palette=self.palette)
        self.draw()

    def action_toggle_list(self):
        self.listing = "functions" if self.listing == "parcels" else "parcels"
        self.refresh_breakdown()

    def on_key(self, event: events.Key):
        if self.mode == "3d" and self.scene:
            step = {"left": (-8, 0), "right": (8, 0), "up": (0, 8), "down": (0, -8)}.get(event.key)
            if step:
                self.scene.rotate(*step)
                self.draw()
        elif self.mode == "slices" and self.slices:
            delta = {"left": -1, "down": -1, "right": 1, "up": 1}.get(event.key)
            if delta:
                self.slices.cursor[2] = max(0, min(self.slices.shape[2] - 1,
                                                   self.slices.cursor[2] + delta))
                self.draw()
            elif event.key in "1234":
                names = list(self.slices.volumes)
                i = int(event.key) - 1
                if i < len(names):
                    self.slices.modality = names[i]
                    self.draw()

    # --------------------------------------------------------------- pipeline

    def action_run(self):
        if self.proc:
            self.log_write("[#f2c14e]already running[/]")
            return
        number = int(self.subject.split("-")[2].split("_")[0])
        self.run_pipeline(number)

    def action_cancel(self):
        if self.proc:
            self.proc.terminate()
            self.log_write("[#f2c14e]cancelled[/]")

    @work(exclusive=True)
    async def run_pipeline(self, number):
        self.log_write(f"[#bfa87f]./main.py {number}[/]")
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
        self.log_write("[#86a186]done[/]")
        self.load_subject(subject_id(number))


if __name__ == "__main__":
    Console().run()
