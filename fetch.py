"""Download UPenn-GBM structural scans from the public TCIA mirror."""

import urllib.request

from paths import INPUTS, MODALITIES, REFERENCE_SEG, modality, raw_dir, subject_id

BASE = "https://github.com/data-nih/tcia/releases/download/upenn-gbm"

# mirror asset suffix -> our modality name
ASSETS = {"T1w.nii.gz": "T1", "ce-gd_T1w.nii.gz": "T1GD",
          "T2w.nii.gz": "T2", "FLAIR.nii.gz": "FLAIR"}


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": "brats-upenn-gbm"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise RuntimeError(f"empty download: {url}")
    tmp.replace(dest)


def get(number, timepoint=1):
    """Fetch one subject's four modalities and reference labels if published."""
    subject = subject_id(number, timepoint)
    prefix = f"sub-{number:03d}" if timepoint == 1 else f"sub-{number:03d}_2"

    for suffix, name in ASSETS.items():
        dest = modality(raw_dir(subject), subject, name)
        # downloads are skipped when present; re-pulling a public mirror each run is wasteful
        if not dest.exists():
            _download(f"{BASE}/{prefix}_{suffix}", dest)

    reference = REFERENCE_SEG / f"{subject}_seg.nii.gz"
    if timepoint == 1 and not reference.exists():
        try:
            _download(f"{BASE}/sub-{number:03d}_seg.nii.gz", reference)
        except Exception:
            pass  # only 147 of 630 subjects have published labels

    missing = [n for n in MODALITIES
               if not modality(raw_dir(subject), subject, n).exists()]
    if missing:
        raise RuntimeError(f"{subject}: missing {', '.join(missing)}")
    return subject
