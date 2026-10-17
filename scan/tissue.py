
import os
import subprocess
from pathlib import Path

from paths import MNI, THREADS, anatomy, out_dir, raw_dir

SYNTHSEG = Path("/opt/freesurfer/bin/mri_synthseg")
FREESURFER_HOME = Path("/opt/freesurfer")


def _shell(args, log, env=None):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(str(a) for a in args)}\n")
        fh.flush()
        result = subprocess.run([str(a) for a in args], stdout=fh,
                                stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({result.returncode}), see {log}")


def run(subject):
    # decision: raw scan on purpose, SynthSeg strips itself. labels inside the lesion aren't trustworthy
    dest = out_dir(subject) / "tissue.nii.gz"
    if dest.exists():
        return dest

    out_dir(subject).mkdir(parents=True, exist_ok=True)
    # explainer: --keepgeom keeps the input grid, the warp below relies on it
    _shell([SYNTHSEG, "--i", anatomy(raw_dir(subject), subject),
            "--o", dest, "--keepgeom", "--cpu", "--threads", THREADS],
           out_dir(subject) / "tissue.log",
           env={**os.environ, "FREESURFER_HOME": str(FREESURFER_HOME)})
    if not dest.exists():
        raise RuntimeError(f"{subject}: synthseg wrote no output")
    return dest


def to_mni(subject):
    dest = out_dir(subject) / "tissue_in_mni.nii.gz"
    if dest.exists():
        return dest

    source, destination = run(subject), out_dir(subject)
    warp, affine = destination / "warp.nii.gz", destination / "affine.mat"
    if not (warp.exists() and affine.exists()):
        raise RuntimeError(f"{subject} is not registered, start from register")

    # decision: LABEL interp, linear invents label values
    _shell(["greedy", "-d", 3, "-rf", MNI, "-ri", "LABEL", "0.2vox",
            "-rm", source, dest, "-r", warp, affine, "-threads", THREADS],
           destination / "tissue.log")
    return dest
