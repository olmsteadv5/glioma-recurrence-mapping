# explainer: RHUH-GBM from its own release, unstripped SRI24 scans run through every stage like UPenn, with RHUH's expert labels kept as the reference

import nibabel as nib
import numpy as np

from paths import MODALITIES, REFERENCE_SEG, RHUH, THREADS, modality, out_dir, raw_dir, stripped_dir, subject_id
from scan.register import _greedy

SOURCE = RHUH / "download" / "PKG - RHUH-GBM-nii-v1" / "RHUH-GBM_nii_v1"
# explainer: RHUH's folders 0, 1, 2 are pre-op, early post-op and recurrence
FOLDERS = {(8, 1): "0", (8, 2): "2", (9, 1): "1", (9, 2): "2"}
NAMES = {"T1": "t1", "T1GD": "t1ce", "T2": "t2", "FLAIR": "flair"}
# explainer: the follow-up has its own grid like UPenn's, the pre-op or post-op scan is aligned to it
FOLLOWUP_ON_PREOP_GRID = False
# decision: SynthSeg's full mode needs more memory than the laptop has on these scans, as with PREDICT
FAST_SYNTHSEG = True
# decision: RHUH ran SynthStrip before registering and z-scoring, a second strip here only trims brain they kept
PRESTRIPPED = True
# explainer: the SRI24 grid RHUH registered every scan to, and the header every scan but two carries
STANDARD = (240, 240, 155)
SRI = np.array([[-1, 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def reorient(image):
    # explainer: an SRI24 scan stored with its axes the other way round, turned back so its voxels and header match the rest
    turn = nib.orientations.ornt_transform(nib.io_orientation(image.affine), nib.io_orientation(SRI))
    data = nib.orientations.apply_orientation(np.asarray(image.dataobj), turn)
    return nib.Nifti1Image(np.ascontiguousarray(data), SRI)


def folder(number, timepoint=1):
    n = number % 10000
    return SOURCE / f"RHUH-{n:04d}" / FOLDERS[number // 10000, timepoint]


def _file(number, timepoint, name):
    where = folder(number, timepoint)
    return where / f"RHUH-{number % 10000:04d}_{where.name}_{name}.nii.gz"


def _mover(number, timepoint, work):
    scan = nib.load(_file(number, timepoint, "t1ce"))
    if scan.shape == STANDARD and np.allclose(scan.affine, SRI, atol=1e-3):
        return nib.load
    if scan.shape == STANDARD:
        # decision: on the SRI24 grid with flipped axes and a header to match, checked against the patient's other scans
        return lambda path, label=False: reorient(nib.load(path))
    # decision: one RHUH scan was never registered to SRI24, so it goes onto the same patient's recurrence scan, which was
    fixed = _file(number, 2, "t1ce")
    matrix = work / "onto_sri.mat"
    _greedy(["-d", 3, "-a", "-dof", 12, "-m", "NMI", "-i", fixed, _file(number, timepoint, "t1ce"), "-o", matrix,
             "-ia-image-centers", "-n", "100x50x20", "-threads", THREADS], work / "onto_sri.log")

    def move(path, label=False):
        dest = work / f"onto_sri_{path.name}"
        # explainer: space outside the scan fills with its own background, a z-scored background is not 0
        fill = ["-ri", "LABEL", "0.2vox"] if label else ["-rb", str(float(np.asarray(nib.load(path).dataobj).min()))]
        _greedy(["-d", 3, "-rf", fixed, *fill, "-rm", path, dest, "-r", matrix, "-threads", THREADS], work / "onto_sri.log")
        moved = nib.load(dest)
        moved = nib.Nifti1Image(np.asarray(moved.dataobj), moved.affine)
        dest.unlink()
        return moved
    return move


def get(number, timepoint=1):
    subject = subject_id(number, timepoint)
    raw_dir(subject).mkdir(parents=True, exist_ok=True)
    load = _mover(number, timepoint, raw_dir(subject))
    images = {}
    for ours, theirs in NAMES.items():
        image = load(_file(number, timepoint, theirs))
        data = np.asarray(image.dataobj, dtype=np.float32)
        # decision: RHUH ships z-scored scans on one constant background, shifted so that background is 0 since every stage here reads > 0 as signal. a shift leaves contrast alone
        images[ours] = data - data.min(), image.affine
    # explainer: RHUH's own brain is every voxel off the background in any of the four, applied to all so they agree
    brain = np.any([data > 0 for data, _ in images.values()], axis=0)
    for d in (stripped_dir(subject), out_dir(subject)):
        d.mkdir(parents=True, exist_ok=True)
    for ours, (data, affine) in images.items():
        nib.save(nib.Nifti1Image(data, affine), modality(raw_dir(subject), subject, ours))
        nib.save(nib.Nifti1Image(np.where(brain, data, 0).astype(np.float32), affine), modality(stripped_dir(subject), subject, ours))
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), images["T1"][1]), out_dir(subject) / "brainmask.nii.gz")
    # decision: adc rides along beside the four sequences, the stages only read those four
    adc = _file(number, timepoint, "adc")
    if adc.exists() and not modality(raw_dir(subject), subject, "ADC").exists():
        nib.save(load(adc), modality(raw_dir(subject), subject, "ADC"))
    # explainer: one recurrence label file is named segmentation.nii.gz with no prefix, so match by the word
    found = sorted(folder(number, timepoint).glob("*segmentation*.nii.gz"))
    labels = found[0] if found else _file(number, timepoint, "segmentations")
    reference = REFERENCE_SEG / f"{subject}_seg.nii.gz"
    if labels.exists() and not reference.exists():
        # decision: rounded to whole labels, a few of RHUH's label files are stored as floats
        image = load(labels, label=True) if load is not nib.load else nib.load(labels)
        nib.save(nib.Nifti1Image(np.rint(np.asarray(image.dataobj)).astype(np.int16), image.affine), reference)
    missing = [m for m in MODALITIES if not modality(raw_dir(subject), subject, m).exists()]
    if missing:
        raise RuntimeError(f"{subject}: missing {', '.join(missing)}")
    return subject
