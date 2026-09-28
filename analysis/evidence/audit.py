import json, pathlib, sys
import nibabel as nib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from paths import raw_dir, stripped_dir, modality, out_dir, MODALITIES, RESULTS
from scan import verify

T = str(pathlib.Path(__file__).parent) + "/"
pairs = json.load(open(T + "pairs41.json"))
size = dict(l.split("\t") for l in open(T + "mirror_sizes.tsv").read().splitlines())
asset = {"T1": "T1w.nii.gz", "T1GD": "ce-gd_T1w.nii.gz", "T2": "T2w.nii.gz", "FLAIR": "FLAIR.nii.gz"}
results = json.load(open(RESULTS))
need = ["seg.nii.gz", "seg_in_mni.nii.gz", "tissue.nii.gz", "tissue_in_mni.nii.gz",
        "affine.mat", "warp.nii.gz", "t1_in_mni.nii.gz", "brainmask.nii.gz"]
algos, bad, rows = set(), [], []
for p in pairs:
    probs = []
    for tp in (1, 2):
        s = f"UPENN-GBM-{p:05d}_{tp}1"
        prefix = f"sub-{p:03d}" + ("" if tp == 1 else "_2")
        for m in MODALITIES:
            f = modality(raw_dir(s), s, m)
            want = int(size.get(f"{prefix}_{asset[m]}", -1))
            if not f.exists(): probs.append(f"{s} raw {m} missing")
            elif f.stat().st_size != want: probs.append(f"{s} raw {m} {f.stat().st_size} B, mirror says {want}")
            if not modality(stripped_dir(s), s, m).exists(): probs.append(f"{s} stripped {m} missing")
        try:
            verify.run(s, stripped_dir(s))
        except Exception as e:
            probs.append(f"{s} verify: {str(e)[:120]}")
        for f in need:
            if not (out_dir(s) / f).exists(): probs.append(f"{s} {f} missing")
        run = (results.get(s) or {}).get("run") or {}
        if not run: probs.append(f"{s} not in results.json")
        algos.add((run.get("algorithm"), run.get("device")))
    try:
        a = nib.load(out_dir(f"UPENN-GBM-{p:05d}_11") / "t1_in_mni.nii.gz").get_fdata() > 0
        b = nib.load(out_dir(f"UPENN-GBM-{p:05d}_21") / "t1_in_mni.nii.gz").get_fdata() > 0
        dice = 2 * (a & b).sum() / (a.sum() + b.sum())
        seg = np.asarray(nib.load(out_dir(f"UPENN-GBM-{p:05d}_11") / "seg_in_mni.nii.gz").dataobj) > 0
        if seg.sum() == 0: probs.append("first scan has no tumor")
    except Exception as e:
        dice = float("nan"); probs.append(f"registration check failed: {str(e)[:80]}")
    if dice < 0.95: probs.append(f"brain dice between timepoints {dice:.3f}")
    rows.append((p, round(float(dice), 4)))
    if probs: bad.append((p, probs))
print(f"{len(pairs)} pairs audited, {len(pairs) - len(bad)} clean")
for p, probs in bad:
    print(f"  {p:05d}:"); [print("    -", x) for x in probs[:8]]
print("segmentation model / device across all scans:", algos)
d = [x[1] for x in rows if x[1] == x[1]]
print(f"brain dice between timepoints: min {min(d):.4f} median {np.median(d):.4f} max {max(d):.4f}")
