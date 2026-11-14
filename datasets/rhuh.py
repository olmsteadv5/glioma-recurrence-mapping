# explainer: RHUH-GBM from its own release, unstripped SRI24 scans run through every stage like UPenn, with RHUH's expert labels kept as the reference

import shutil

import nibabel as nib
import numpy as np

from paths import MODALITIES, REFERENCE_SEG, RHUH, modality, out_dir, raw_dir, stripped_dir, subject_id

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


def folder(number, timepoint=1):
    n = number % 10000
    return SOURCE / f"RHUH-{n:04d}" / FOLDERS[number // 10000, timepoint]


def _file(number, timepoint, name):
    where = folder(number, timepoint)
    return where / f"RHUH-{number % 10000:04d}_{where.name}_{name}.nii.gz"


def get(number, timepoint=1):
    subject = subject_id(number, timepoint)
    raw_dir(subject).mkdir(parents=True, exist_ok=True)
    images = {}
    for ours, theirs in NAMES.items():
        image = nib.load(_file(number, timepoint, theirs))
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
        shutil.copyfile(adc, modality(raw_dir(subject), subject, "ADC"))
    # explainer: one recurrence label file is named segmentation.nii.gz with no prefix, so match by the word
    found = sorted(folder(number, timepoint).glob("*segmentation*.nii.gz"))
    labels = found[0] if found else _file(number, timepoint, "segmentations")
    reference = REFERENCE_SEG / f"{subject}_seg.nii.gz"
    if labels.exists() and not reference.exists():
        # decision: rounded to whole labels, a few of RHUH's label files are stored as floats
        image = nib.load(labels)
        nib.save(nib.Nifti1Image(np.rint(np.asarray(image.dataobj)).astype(np.int16), image.affine), reference)
    missing = [m for m in MODALITIES if not modality(raw_dir(subject), subject, m).exists()]
    if missing:
        raise RuntimeError(f"{subject}: missing {', '.join(missing)}")
    return subject
