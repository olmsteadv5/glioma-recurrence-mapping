import csv, json, sys
from pathlib import Path
import nibabel as nib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import OUTPUTS, modality, out_dir, stripped_dir

pairs = json.load(open(Path(__file__).parent / "pairs41.json"))
meta = {r["ID"]: r for r in csv.DictReader(open(OUTPUTS / "participants.tsv"), delimiter="\t")}
COLORS = {1: "#ffd21f", 3: "#ff3b30", 4: "#ff4fd8"}  # NETC, ET, RC

def window(img, mask):
    v = img[mask]; lo, hi = np.percentile(v, [1, 99.5]); return np.clip((img - lo) / (hi - lo), 0, 1)

def sheet(batch, path):
    fig, axes = plt.subplots(len(batch), 3, figsize=(9, 3.1 * len(batch)))
    for row, n in zip(axes, batch):
        first, later = f"UPENN-GBM-{n:05d}_11", f"UPENN-GBM-{n:05d}_21"
        core0 = np.isin(np.asarray(nib.load(out_dir(first) / "seg.nii.gz").dataobj), (1, 3))
        seg = np.asarray(nib.load(out_dir(later) / "seg.nii.gz").dataobj)
        brain = np.asarray(nib.load(out_dir(later) / "brainmask.nii.gz").dataobj) > 0
        z = int(np.argmax(core0.sum(axis=(0, 1))))
        for ax, m in zip(row, ("T1GD", "FLAIR", "T2")):
            img = np.asarray(nib.load(modality(stripped_dir(later), later, m)).dataobj, dtype=np.float32)
            ax.imshow(np.rot90(window(img, brain)[:, :, z]), cmap="gray")
            ax.contour(np.rot90(core0[:, :, z]), levels=[0.5], colors="#00e0ff", linewidths=0.9, linestyles="dashed")
            for v, c in COLORS.items():
                if (seg[:, :, z] == v).any():
                    ax.contour(np.rot90(seg[:, :, z] == v), levels=[0.5], colors=c, linewidths=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(m, fontsize=8)
        r = meta[f"UPENN-GBM-{n:05d}_21"]; g = meta[f"UPENN-GBM-{n:05d}_11"]["GTR_over90percent"]
        row[0].set_ylabel(f"{n:05d}\npath {r['PsP_TP_score']}  GTR {g}\n{r['Time_since_baseline_preop']} d", fontsize=8)
    fig.text(0.01, 0.002, "cyan dashed = original tumor core   yellow = NETC   red = ET   magenta = RC", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

(OUTPUTS / "figures" / "cavity_review").mkdir(parents=True, exist_ok=True)
for i in range(0, len(pairs), 7):
    sheet(pairs[i:i + 7], OUTPUTS / "figures" / "cavity_review" / f"cav_{i // 7}.png")
print("sheets written")
