# explainer: brings an outside dataset's stripped, segmented scans into the same files the UPenn stages would have made

import shutil

import nibabel as nib

from datasets import adapter
from scan.register import _greedy, _metric_mask
from paths import MNI, THREADS, modality, out_dir, pair_dir, raw_dir, stripped_dir, subject_id


def run(number, timepoint):
    data = adapter(number)
    subject = subject_id(number, timepoint)
    destination = out_dir(subject)
    for d in (raw_dir(subject), stripped_dir(subject), destination):
        d.mkdir(parents=True, exist_ok=True)
    log = destination / "ingest.log"
    log.write_text(f"{data.__name__} {data.folder(number)}\n")

    for name, path in data.scans(number, timepoint).items():
        nib.save(data.load(path, number), modality(stripped_dir(subject), subject, name))
        shutil.copyfile(modality(stripped_dir(subject), subject, name), modality(raw_dir(subject), subject, name))
    nib.save(data.load(data.brainmask(number), number, is_labels=True), destination / "brainmask.nii.gz")

    # decision: keep the dataset's own labels as seg_auto even when expert labels win, so the two can be compared
    seg, auto = destination / "seg.nii.gz", destination / "seg_auto.nii.gz"
    nib.save(data.load(data.labels(number, timepoint), number, is_labels=True), auto)
    fixed = modality(stripped_dir(subject), subject, "T1GD")
    if data.expert(number, timepoint, fixed, seg, log):
        return subject, "expert"
    shutil.copyfile(auto, seg)
    return subject, "auto"


def reuse_registration(later, seg_path):
    # decision: a follow-up already on the pre-op grid takes the pre-op scan's MNI warp instead of its own
    first = later[:-2] + "11"
    destination = out_dir(later)
    for name in ("affine.mat", "warp.nii.gz", "warp_inverse.nii.gz"):
        shutil.copyfile(out_dir(first) / name, destination / name)
    t1 = modality(stripped_dir(later), later, "T1GD")
    kept = _metric_mask(t1, seg_path, destination / "metric_mask.nii.gz")
    seg_mni = destination / "seg_in_mni.nii.gz"
    _greedy(["-d", 3, "-rf", MNI, "-rm", t1, destination / "t1_in_mni.nii.gz",
             "-ri", "LABEL", "0.2vox", "-rm", seg_path, seg_mni,
             "-r", destination / "warp.nii.gz", destination / "affine.mat", "-threads", THREADS],
            destination / "registration.log")
    return seg_mni, kept


def identity_align(later):
    # decision: same grid already, so align is an identity and the pre-op seg goes over as is
    first = later[:-2] + "11"
    destination = pair_dir(later)
    (destination / "rigid_first_to_later.mat").write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    shutil.copyfile(out_dir(first) / "seg.nii.gz", destination / "first_seg_in_later.nii.gz")
    shutil.copyfile(modality(stripped_dir(first), first, "T1GD"), destination / "first_t1_in_later.nii.gz")
    (destination / "align.log").write_text("follow-up already on the pre-op grid, identity\n")
