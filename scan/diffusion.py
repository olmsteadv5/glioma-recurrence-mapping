#!/home/vilem/Projects/brats-upenn-gbm/.venv/bin/python
# explainer: tractography from each UPenn patient's own pre-op diffusion scan, put on the pre-op grid with no MNI step

import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hcp.streamlines import _resample
from scan.register import _greedy, _metric_mask
from paths import (DCM2NIIX, DMRI_ENV, DSI_STUDIO, FERNET, FREESURFER_LICENSE, INPUTS, PAIRS, SYNB0, THREADS, modality,
                   out_dir, stripped_dir, subject_id)

DTI = INPUTS / "upenn_dti"
PREP = INPUTS / "upenn_dti_prep"
SERIES = DTI / "series.json"
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID="
# decision: same streamline count as the atlas walk-down uses, so the two compare like for like
TRACTS = 48000


# explainer: set by --prep, then every stage reads the corrected scan and writes beside the uncorrected one
PREPPED = False


def raw(subject):
    return (PREP if PREPPED else DTI) / subject


def work(subject):
    path = out_dir(subject) / ("diffusion_prep" if PREPPED else "diffusion")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dsi(args, log):
    with open(log, "a") as fh:
        fh.write(f"\n$ dsi_studio {' '.join(args)}\n")
        fh.flush()
        done = subprocess.run([str(DSI_STUDIO), *args], stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    if done.returncode != 0:
        raise RuntimeError(f"dsi_studio {args[0]} failed, see {log}")


def pick(subject):
    # decision: pre-op is the earliest study and follow-up the latest, the plain series beats _T / _TERA / ONLY re-sends, then the one with the most volumes wins
    pid = subject[:15]
    mine = [s for s in json.loads(SERIES.read_text()) if s["PatientID"] == pid]
    when = (min if subject.endswith("_11") else max)(s["StudyDate"][:10] for s in mine)
    found = [s for s in mine if s["StudyDate"][:10] == when
             and re.search(r"DTI|DIFF|MDDW", s.get("SeriesDescription") or "", re.I)]
    found.sort(key=lambda s: (bool(re.search(r"_T\b|_TERA|ONLY", s["SeriesDescription"], re.I)), -s["ImageCount"]))
    return found[0] if found else None


def _env(args, log):
    env = {**os.environ, "PATH": f"{DMRI_ENV / 'bin'}:{os.environ['PATH']}", "FSLDIR": str(DMRI_ENV),
           "FSLOUTPUTTYPE": "NIFTI_GZ", "PYTHONPATH": str(FERNET), "TMPDIR": os.environ.get("TMPDIR", "/tmp")}
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(str(a) for a in args)}\n")
        fh.flush()
        done = subprocess.run([str(a) for a in args], stdout=fh, stderr=subprocess.STDOUT, env=env)
    if done.returncode != 0:
        raise RuntimeError(f"{args[0]} failed, see {log}")


def _to_dwi(subject, image, ref, dest, log, label=False):
    # explainer: a pre-op structural image into the diffusion scan's own space, through the inverse of the checked diffusion to T2 fit
    back = out_dir(subject) / "diffusion" / "to_t2.mat"
    _greedy(["-d", 3, "-rf", ref, *(["-ri", "LABEL", "0.2vox"] if label else []), "-rm", image, dest,
             "-r", f"{back},-1", "-threads", THREADS], log)
    return np.asarray(nib.load(dest).dataobj)


def _save(data, image, dest):
    # explainer: keep both qform and sform, synb0's FSL steps read the qform
    out = nib.Nifti1Image(data, image.affine)
    out.set_qform(image.affine, 1)
    out.set_sform(image.affine, 1)
    nib.save(out, dest)


def preprocess(subject):
    # explainer: denoise, unring, then motion, eddy and distortion correction, the raw scan has no reverse phase-encode so synb0 stands in
    source, d = DTI / subject, PREP / subject
    if (d / "dwi.nii.gz").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    log = d / "prep.log"
    threads = os.environ.get("DMRI_THREADS", "12")
    # decision: denoising has to come first, before anything interpolates the noise
    if not (d / "unrung.nii.gz").exists():
        _env(["dwidenoise", source / "dwi.nii.gz", d / "denoised.nii.gz", "-noise", d / "noise.nii.gz", "-nthreads", threads, "-force"], log)
        _env(["mrdegibbs", d / "denoised.nii.gz", d / "unrung.tmp.nii.gz", "-nthreads", threads, "-force"], log)
        os.replace(d / "unrung.tmp.nii.gz", d / "unrung.nii.gz")
    bvals = np.loadtxt(source / "dwi.bval")
    readout = json.loads((source / "dwi.json").read_text())["TotalReadoutTime"]
    image = nib.load(d / "unrung.nii.gz")
    _save(np.asarray(image.dataobj)[..., bvals < 50].mean(-1).astype(np.float32), image, d / "b0.nii.gz")
    # decision: UPenn's T1 sits in SRI24 space far from the scanner frame, so it goes into the b0's frame first or synb0 misregisters
    zooms = np.array(image.header.get_zooms()[:3])
    fine = image.affine.copy()
    fine[:3, :3] = image.affine[:3, :3] / zooms
    grid = nib.Nifti1Image(np.zeros(np.ceil(np.array(image.shape[:3]) * zooms).astype(int), np.float32), fine)
    _save(grid.get_fdata(dtype=np.float32), grid, d / "grid_1mm.nii.gz")
    inputs, outputs = d / "synb0_in", d / "synb0_out"
    inputs.mkdir(exist_ok=True)
    if not (inputs / "acqparams.txt").exists():
        t1 = _to_dwi(subject, modality(stripped_dir(subject), subject, "T1"), d / "grid_1mm.nii.gz", inputs / "T1.nii.gz", log)
        _save(t1.astype(np.float32), nib.load(d / "grid_1mm.nii.gz"), inputs / "T1.nii.gz")
        shutil.copyfile(d / "b0.nii.gz", inputs / "b0.nii.gz")
        # decision: the sign of the phase-encode axis isn't in the header, either sign works since topup and eddy read the same line
        (inputs / "acqparams.txt").write_text(f"0 1 0 {readout}\n0 1 0 0\n")
    if not (outputs / "topup_movpar.txt").exists():
        # decision: synb0's cpu inference needs more memory than the laptop spares, so by default it runs on the cluster from synb0_in
        if os.environ.get("SYNB0_LOCAL") != "1":
            raise RuntimeError(f"{subject}: synb0 inputs are ready in {inputs}, run tools/bouchet/synb0.sbatch and copy synb0_out back")
        shutil.rmtree(outputs, ignore_errors=True)
        outputs.mkdir()
        _env(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
              "-v", f"{inputs}:/INPUTS/", "-v", f"{outputs}:/OUTPUTS/", "-v", f"{FREESURFER_LICENSE}:/extra/freesurfer/license.txt",
              SYNB0, "--stripped"], log)
    # decision: the synthetic b0 should need only a small shift onto the real one, a big one means synb0 misregistered
    moved = np.loadtxt(outputs / "topup_movpar.txt")[1]
    if np.linalg.norm(moved[:3]) > 5 or np.abs(moved[3:]).max() > 0.1:
        raise RuntimeError(f"{subject}: synb0 and the real b0 are {np.linalg.norm(moved[:3]):.1f} mm apart, see {outputs}")
    # decision: UPenn's own brain mask moved into diffusion space, bet clips tumors and cavities
    brain = _to_dwi(subject, out_dir(subject) / "brainmask.nii.gz", d / "b0.nii.gz", d / "brain_mask.nii.gz", log, label=True) > 0
    brain = ndimage.binary_dilation(brain, iterations=1)
    ml = brain.sum() * np.prod(zooms) / 1000
    if not 800 < ml < 2200:
        raise RuntimeError(f"{subject}: brain mask in diffusion space is {ml:.0f} mL, see {d}")
    _save(brain.astype(np.uint8), image, d / "brain_mask.nii.gz")
    tumor = _to_dwi(subject, out_dir(subject) / "seg.nii.gz", d / "b0.nii.gz", d / "tumor_mask.nii.gz", log, label=True) > 0
    _save(ndimage.binary_dilation(tumor, iterations=2).astype(np.uint8), image, d / "tumor_mask.nii.gz")
    (d / "acqp.txt").write_text(f"0 1 0 {readout}\n")
    (d / "index.txt").write_text(" ".join(["1"] * len(bvals)) + "\n")
    if not (d / "eddy.nii.gz").exists() and os.environ.get("EDDY_LOCAL") != "1":
        # decision: eddy is slow on the laptop cpu and quick on a cluster gpu, so by default it runs there
        raise RuntimeError(f"{subject}: eddy inputs are ready in {d}, run tools/bouchet/eddy.sbatch and copy eddy.* back")
    # decision: linear second-level model for a half-sphere scheme with few directions, replace slices that lost signal, and smooth the first pass for patients who move
    if not (d / "eddy.nii.gz").exists():
        _env(["eddy_cpu", f"--imain={d / 'unrung.nii.gz'}", f"--mask={d / 'brain_mask.nii.gz'}", f"--acqp={d / 'acqp.txt'}",
              f"--index={d / 'index.txt'}", f"--bvecs={source / 'dwi.bvec'}", f"--bvals={source / 'dwi.bval'}",
              f"--topup={outputs / 'topup'}", f"--out={d / 'eddy'}", "--repol", "--slm=linear", "--fwhm=10,0,0,0,0",
              "--cnr_maps", f"--nthr={threads}"], log)
    shutil.copyfile(d / "eddy.eddy_rotated_bvecs", d / "dwi.bvec")
    shutil.copyfile(source / "dwi.bval", d / "dwi.bval")
    shutil.copyfile(source / "dwi.json", d / "dwi.json")
    # decision: eddy's spline can undershoot below zero at edges, a negative signal has no meaning and breaks the free water fit
    corrected = nib.load(d / "eddy.nii.gz")
    _save(np.clip(np.asarray(corrected.dataobj), 0, None).astype(np.float32), corrected, d / "eddy_clipped.nii.gz")
    os.replace(d / "eddy_clipped.nii.gz", d / "dwi.nii.gz")
    (d / "eddy.nii.gz").unlink()


def free_water(subject):
    # explainer: fernet splits each voxel into tissue and free water, single shell works because of its interpolated start
    d = PREP / subject
    out = d / "fernet"
    if (out / "corrected_fw_DWI.nii.gz").exists():
        return out
    out.mkdir(exist_ok=True)
    python, log = DMRI_ENV / "bin" / "python", d / "fernet.log"
    # decision: fernet divides by b0, so voxels with no b0 signal stay out of its mask or they come back as NaN
    image = nib.load(d / "dwi.nii.gz")
    b0 = np.asarray(image.dataobj)[..., np.loadtxt(d / "dwi.bval") < 50].mean(-1)
    brain = np.asarray(nib.load(d / "brain_mask.nii.gz").dataobj) > 0
    _save((brain & (b0 > 0)).astype(np.uint8), image, d / "fernet_mask.nii.gz")
    grads = ["-r", d / "dwi.bvec", "-b", d / "dwi.bval", "-m", d / "fernet_mask.nii.gz"]
    # decision: keep tumor, edema and cavity out of the reference white matter and CSF it picks
    _env([python, FERNET / "scripts" / "fernet.py", "-d", d / "dwi.nii.gz", *grads, "-x", d / "tumor_mask.nii.gz",
          "-o", out / "fernet", "-B"], log)
    _env([python, FERNET / "scripts" / "fernet_fw_dwi.py", "-d", d / "dwi.nii.gz", *grads,
          "-f", out / "fernet_fw_volume_fraction.nii.gz", "-o", out / "corrected"], log)
    return out


def fetch(subject):
    d = raw(subject)
    if (d / "dwi.fz").exists():
        return True
    if PREPPED:
        if not (DTI / subject / "dwi.nii.gz").exists():
            return False
        preprocess(subject)
        free_water(subject)
        return _reconstruct(subject)
    series = pick(subject)
    if series is None:
        return False
    d.mkdir(parents=True, exist_ok=True)
    log = d / "fetch.log"
    if not (d / "dwi.nii.gz").exists():
        archive = d / "series.zip"
        for _ in range(4):
            if (subprocess.run(["curl", "-sf", "--max-time", "900", "-o", str(archive), API + series["SeriesInstanceUID"]]).returncode == 0
                    and zipfile.is_zipfile(archive)):
                break
            time.sleep(10)
        else:
            raise RuntimeError(f"{subject}: download failed")
        zipfile.ZipFile(archive).extractall(d / "dicom")
        archive.unlink()
        subprocess.run([str(DCM2NIIX), "-z", "y", "-f", "dwi", "-o", str(d), str(d / "dicom")],
                       stdout=open(log, "a"), stderr=subprocess.STDOUT, check=True)
    return _reconstruct(subject)


def _reconstruct(subject):
    d = raw(subject)
    log = d / "fetch.log"
    _dsi(["--action=src", f"--source={d}/dwi.nii.gz", f"--bval={d}/dwi.bval", f"--bvec={d}/dwi.bvec", f"--output={d}/dwi.sz"], log)
    # explainer: qc writes qc.tsv into the working directory unless told otherwise
    _dsi(["--action=qc", f"--source={d}/dwi.sz", f"--output={d}/qc.tsv"], log)
    # decision: check_btable picks the gradient axis order on the raw scan, the corrected scan keeps eddy's rotated table as is
    _dsi(["--action=rec", f"--source={d}/dwi.sz", "--method=4", "--param=1.25", f"--check_btable={0 if PREPPED else 1}",
          "--make_isotropic=2", "--bias_field_correction=1", "--motion_correction=0",
          f"--thread_count={THREADS}", f"--output={d}/dwi.fz"], log)
    return True


def dsi_to_world(subject):
    # explainer: dsi resamples to its own 2 mm grid and header, so map its voxels onto the scanner grid, flipping any axis that disagrees
    d = raw(subject)
    scan, grid = nib.load(d / "dwi.nii.gz"), nib.load(d / "dwi.fz.iso.nii.gz")
    zoom, size = np.array(scan.header.get_zooms()[:3]), np.array(scan.shape[:3])
    step = np.abs(np.diag(grid.affine)[:3])
    flip = np.sign(np.diag(grid.affine)[:3]) != np.sign(np.diag(scan.affine)[:3])
    to_scan = np.eye(4)
    for axis in range(3):
        to_scan[axis, axis] = (-1 if flip[axis] else 1) * step[axis] / zoom[axis]
        to_scan[axis, 3] = size[axis] - 1 if flip[axis] else 0
    corrected = scan.affine @ to_scan
    return corrected @ np.linalg.inv(grid.affine), corrected


def _silhouette(path, dest):
    image = nib.load(path)
    brain = ndimage.binary_fill_holes(ndimage.binary_closing(np.asarray(image.dataobj) > 0, iterations=2))
    nib.save(nib.Nifti1Image(ndimage.gaussian_filter(brain.astype(np.float32), 2 / np.array(image.header.get_zooms()[:3])), image.affine), dest)


def _rotation(matrix):
    m = np.loadtxt(matrix)[:3, :3]
    m = m / np.cbrt(abs(np.linalg.det(m)))
    return np.degrees(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)))


def _fa_contrast(subject, matrix, dest):
    # explainer: fa belongs in white matter, so a good fit puts high fa on SynthSeg's white matter and low fa on cortex
    w = work(subject)
    _greedy(["-d", 3, "-rf", modality(stripped_dir(subject), subject, "T2"), "-rm", w / "fa.nii.gz", dest, "-r", matrix,
             "-threads", THREADS], w / "register.log")
    fa = np.asarray(nib.load(dest).dataobj)
    tissue = np.asarray(nib.load(out_dir(subject) / "tissue.nii.gz").dataobj)
    return float(fa[np.isin(tissue, (2, 41))].mean() - fa[np.isin(tissue, (3, 42))].mean())


def register(subject, tries=4):
    # decision: dsi's iso map looks most like a T2, so it goes onto the same session's T2. rigid then affine, the affine soaks up some EPI distortion
    w = work(subject)
    if (w / "to_t2.mat").exists():
        return w / "to_t2.mat"
    d = raw(subject)
    _dsi(["--action=exp", f"--source={d}/dwi.fz", "--export=iso,fa"], w / "register.log")
    _, corrected = dsi_to_world(subject)
    for name in ("iso", "fa"):
        image = nib.load(d / f"dwi.fz.{name}.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj).astype(np.float32), corrected), w / f"{name}.nii.gz")
    t2 = modality(stripped_dir(subject), subject, "T2")
    _metric_mask(t2, out_dir(subject) / "seg.nii.gz", w / "mask.nii.gz")
    # decision: start from the two brain silhouettes, that fit is smooth and lands the same every run
    _silhouette(t2, w / "t2_shape.nii.gz")
    _silhouette(w / "iso.nii.gz", w / "iso_shape.nii.gz")
    _greedy(["-d", 3, "-a", "-dof", 6, "-m", "SSD", "-i", w / "t2_shape.nii.gz", w / "iso_shape.nii.gz",
             "-ia-image-centers", "-o", w / "shape.mat", "-n", "100x100x50", "-threads", THREADS], w / "register.log")
    # decision: NMI refine only at the two finest levels (coarse ones jumped up to 90 degrees), a few runs each, keep the fit with fa most on white matter
    start, best = w / "shape.mat", None
    # decision: once distortion is corrected the fit stays rigid, an affine would only soak up anatomy
    for dof in (6,) if PREPPED else (6, 12):
        scored = [(_fa_contrast(subject, start, w / "fa_try.nii.gz"), start)]
        for k in range(tries):
            out = w / f"try_{dof}_{k}.mat"
            _greedy(["-d", 3, "-a", "-dof", dof, "-m", "NMI", "-i", t2, w / "iso.nii.gz", "-gm", w / "mask.nii.gz",
                     "-ia", start, "-o", out, "-n", "20x50", "-threads", THREADS], w / "register.log")
            if _rotation(out) <= 30:
                scored.append((_fa_contrast(subject, out, w / "fa_try.nii.gz"), out))
        best = max(scored)
        start = best[1]
    if best[0] <= 0:
        raise RuntimeError(f"{subject}: no fit puts fa on white matter, see {w / 'register.log'}")
    shutil.copyfile(best[1], w / "to_t2.mat")
    for leftover in list(w.glob("try_*.mat")) + [w / "fa_try.nii.gz"]:
        leftover.unlink(missing_ok=True)
    _fa_contrast(subject, w / "to_t2.mat", w / "fa_on_t2.nii.gz")
    return w / "to_t2.mat"


# explainer: dsi maps live on dsi's grid, fernet's and the plain tensor's on the corrected scan's own grid
DSI_MAPS = ("qa", "md")
# decision: fernet's trace is three diffusivities summed in mm2/s, so a third of it in dsi's 1e-3 units is the tissue md
FERNET_MAPS = {"fw": ("fernet_fw_volume_fraction", 1.0), "tissue_md": ("fernet_fw_tensor_TR", 1000 / 3),
               "tissue_fa": ("fernet_fw_tensor_FA", 1.0), "tissue_ad": ("fernet_fw_tensor_AX", 1000.0),
               "tissue_rd": ("fernet_fw_tensor_RAD", 1000.0)}
# decision: a plain tensor fit on fernet's grid and mask, so corrected vs uncorrected differ only by the free water
TENSOR_MAPS = {"dti_fa": ("-fa", 1.0), "dti_md": ("-adc", 1000.0)}
# decision: below this share of a voxel inside the diffusion field of view the value is mostly made up
COVERED = 0.5


def _plain_tensor(d, log):
    if all((d / f"{name}.nii.gz").exists() for name in TENSOR_MAPS):
        return
    _env(["dwi2tensor", d / "dwi.nii.gz", d / "dti_tensor.nii.gz", "-fslgrad", d / "dwi.bvec", d / "dwi.bval",
          "-mask", d / "fernet_mask.nii.gz", "-force", "-quiet"], log)
    for name, (flag, _) in TENSOR_MAPS.items():
        _env(["tensor2metric", d / "dti_tensor.nii.gz", flag, d / f"{name}.nii.gz", "-force", "-quiet"], log)
    (d / "dti_tensor.nii.gz").unlink()


def maps(subject):
    # explainer: every scalar map of the corrected scan onto the same session's T2 grid through the checked diffusion to T2 fit
    w, d = work(subject), raw(subject)
    t2 = modality(stripped_dir(subject), subject, "T2")
    log = w / "register.log"
    brain = np.asarray(nib.load(out_dir(subject) / "brainmask.nii.gz").dataobj) > 0
    names = [*DSI_MAPS, *FERNET_MAPS, *TENSOR_MAPS]
    outputs = [w / f"{name}_on_t2.nii.gz" for name in ("coverage", *names)]
    # decision: a refit of the diffusion to T2 transform makes every older map stale
    fitted = (w / "to_t2.mat").stat().st_mtime
    if all(path.exists() and path.stat().st_mtime > fitted for path in outputs):
        return outputs
    _dsi(["--action=exp", f"--source={d}/dwi.fz", f"--export={','.join(DSI_MAPS)}"], log)
    _, corrected = dsi_to_world(subject)
    sources = {}
    for name in DSI_MAPS:
        image = nib.load(d / f"dwi.fz.{name}.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj).astype(np.float32), corrected), w / f"{name}.nii.gz")
        sources[name] = (w / f"{name}.nii.gz", 1.0)
    _plain_tensor(d, log)
    sources.update({name: (d / "fernet" / f"{stem}.nii.gz", scale) for name, (stem, scale) in FERNET_MAPS.items()})
    sources.update({name: (d / f"{name}.nii.gz", scale) for name, (_, scale) in TENSOR_MAPS.items()})
    tmp = w / "resampled.tmp.nii.gz"

    def resample(source):
        _greedy(["-d", 3, "-rf", t2, "-rm", source, tmp, "-r", w / "to_t2.mat", "-threads", THREADS], log)
        moved = nib.load(tmp)
        return np.asarray(moved.dataobj), moved.affine

    # explainer: written to a temp name and moved into place only once final, so a crash never leaves a raw map that looks done
    def save(data, affine, dest):
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine), tmp)
        os.replace(tmp, dest)

    # explainer: share of each T2 voxel that came from inside the diffusion mask, the analysis picks its own cutoff
    values, affine = resample(d / "fernet_mask.nii.gz")
    share = np.where(brain, values, 0)
    save(share, affine, outputs[0])
    for name, (source, scale) in sources.items():
        values, affine = resample(source)
        # decision: nan outside the brain and the diffusion field of view, a zero there would pass for a real value in smoothing
        save(np.where(brain & (share >= COVERED), values * scale, np.nan), affine, w / f"{name}_on_t2.nii.gz")
    return outputs


def track(subject):
    w = work(subject)
    if not (w / "tracts.trk.gz").exists():
        # decision: whole brain, not tumor seeded, since fa drops in the edema and seeds there stop early
        _dsi(["--action=trk", f"--source={raw(subject)}/dwi.fz", f"--tract_count={TRACTS}", "--turning_angle=60",
              "--min_length=30", "--max_length=300", "--random_seed=1", f"--thread_count={THREADS}",
              f"--output={w}/tracts.trk.gz"], w / "track.log")
    return w / "tracts.trk.gz"


def to_native(subject):
    # explainer: trk points sit in dsi's frame. dsi -> scanner -> T2 grid, the last step through greedy's inverse since points move opposite to images
    w = work(subject)
    lines = [_resample(np.asarray(line, float)) for line in nib.streamlines.load(track(subject)).streamlines]
    offsets = np.cumsum([0] + [len(line) for line in lines])
    to_world, _ = dsi_to_world(subject)
    points = nib.affines.apply_affine(to_world, np.vstack(lines)).astype(np.float32)
    source, dest = w / "points.vtk", w / "points_on_t2.vtk"
    with open(source, "wb") as fh:
        fh.write(f"# vtk DataFile Version 3.0\nstreamlines\nBINARY\nDATASET POLYDATA\nPOINTS {len(points)} float\n".encode())
        fh.write(points.astype(">f4").tobytes())
        fh.write(b"\n")
    _greedy(["-d", 3, "-rf", modality(stripped_dir(subject), subject, "T2"), "-rs", source, dest,
             "-r", f"{w / 'to_t2.mat'},-1", "-threads", THREADS], w / "register.log")
    text = dest.read_text().split("POINTS", 1)[1].split("\n", 1)[1]
    moved = np.array(text.split()[:3 * len(points)], dtype=np.float32).reshape(-1, 3)
    source.unlink()
    dest.unlink()
    np.savez(w / "tracts_native.npz", points=moved, offsets=offsets)
    return moved


def check(subject, points):
    # decision: tracts belong in white matter, so the share of points landing there is the registration check
    t2 = nib.load(modality(stripped_dir(subject), subject, "T2"))
    tissue = np.asarray(nib.load(out_dir(subject) / "tissue.nii.gz").dataobj)
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(t2.affine), points)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < t2.shape), axis=1)
    label = tissue[tuple(ijk[inside].T)]
    fa = np.asarray(nib.load(work(subject) / "fa_on_t2.nii.gz").dataobj)
    result = {"white_matter": round(float(np.isin(label, (2, 41)).mean()), 3),
              "cortex": round(float(np.isin(label, (3, 42)).mean()), 3),
              "outside_brain": round(float((label == 0).mean() + (1 - inside.mean())), 3),
              "fa_white_matter": round(float(fa[np.isin(tissue, (2, 41))].mean()), 3),
              "fa_cortex": round(float(fa[np.isin(tissue, (3, 42))].mean()), 3)}
    (work(subject) / "check.json").write_text(json.dumps(result, indent=1))
    return result


def run(number, timepoint=1):
    subject = subject_id(number, timepoint)
    if not fetch(subject):
        return subject, "no diffusion at this timepoint"
    register(subject)
    if PREPPED:
        maps(subject)
    return subject, check(subject, to_native(subject))


if __name__ == "__main__":
    timepoint = 2 if "--follow-up" in sys.argv else 1
    PREPPED = "--prep" in sys.argv
    numbers = [int(a) for a in sys.argv[1:] if a.isdigit()] or [int(p) for p in sorted(os.listdir(PAIRS)) if p.isdigit() and int(p) < 10000]
    for n in numbers:
        try:
            print(*run(n, timepoint), flush=True)
        except Exception as error:
            print(subject_id(n, timepoint), "failed:", error, flush=True)
