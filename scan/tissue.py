# Tissue labels from SynthSeg; nothing upstream tells white matter from CSF.

import os
import subprocess
from pathlib import Path

from paths import MNI, THREADS, modality, out_dir, raw_dir

SYNTHSEG = Path("/opt/freesurfer/bin/mri_synthseg")
FREESURFER_HOME = Path("/opt/freesurfer")

# FreeSurfer label numbers, left then right where the structure is paired.
WM = (2, 41)
CORTEX = (3, 42)
VENTRICLE = (4, 5, 43, 44)
MIDLINE_VENTRICLE = (14, 15)
CSF = (24,)
CEREBELLUM = (7, 8, 46, 47)
BRAINSTEM = (16,)
HIPPOCAMPUS = (17, 53)
LEFT = (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28)
RIGHT = (41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60)


def _shell(args, log, env=None):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(str(a) for a in args)}\n")
        fh.flush()
        result = subprocess.run([str(a) for a in args], stdout=fh,
                                stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}), see {log}")


def run(subject):
    # why: SynthSeg skull-strips and normalizes internally, so it wants the raw
    #      scan and the stripped one would only take tissue away [billot23]
    # Labels inside the lesion are unvalidated: on 00055_11 SynthSeg fills the
    # mass with white matter and a slab of cortex 12 mm deep.
    dest = out_dir(subject) / "tissue.nii.gz"
    if dest.exists():
        return dest

    out_dir(subject).mkdir(parents=True, exist_ok=True)
    # --keepgeom holds the labels on the input grid, which the warp below assumes
    _shell([SYNTHSEG, "--i", modality(raw_dir(subject), subject, "T1"),
            "--o", dest, "--keepgeom", "--cpu", "--threads", THREADS],
           out_dir(subject) / "tissue.log",
           env={**os.environ, "FREESURFER_HOME": str(FREESURFER_HOME)})
    if not dest.exists():
        raise RuntimeError(f"{subject}: synthseg wrote no output")
    return dest


def to_mni(subject):
    # The same reslice chain register.py uses, since the grid is the same.
    dest = out_dir(subject) / "tissue_in_mni.nii.gz"
    if dest.exists():
        return dest

    source, destination = run(subject), out_dir(subject)
    warp, affine = destination / "warp.nii.gz", destination / "affine.mat"
    if not (warp.exists() and affine.exists()):
        raise RuntimeError(f"{subject} is not registered, start from register")

    # LABEL, never linear: interpolating label numbers invents structures
    _shell(["greedy", "-d", 3, "-rf", MNI, "-ri", "LABEL", "0.2vox",
            "-rm", source, dest, "-r", warp, affine, "-threads", THREADS],
           destination / "tissue.log")
    return dest
