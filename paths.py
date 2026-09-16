# Shared paths and subject-name helpers.

from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
REFERENCE_SEG = INPUTS / "reference_seg"
ATLAS_INPUT = INPUTS / "atlas"
ATLAS_DIR = OUTPUTS / "atlas"
RESULTS = OUTPUTS / "results.json"

MNI = Path("/home/vilem/Documents/SeizureAnalysis/from_downloads/MNI152_T1_1mm_Brain.nii.gz")
SYNTHSTRIP = Path("/opt/freesurfer/bin/mri_synthstrip")

MODALITIES = ("T1", "T1GD", "T2", "FLAIR")
LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
REGIONS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3), "RC": (4,)}

THREADS = "12"


def subject_id(number, timepoint=1):
    return f"UPENN-GBM-{number:05d}_{timepoint}1"


def raw_dir(subject):
    return INPUTS / subject


def out_dir(subject):
    return OUTPUTS / subject


def stripped_dir(subject):
    return out_dir(subject) / "stripped"


def modality(directory, subject, name):
    return directory / f"{subject}_{name}.nii.gz"


STAGES = ("fetch", "strip", "segment", "register", "map")


def stage_file(subject, stage):
    # The file whose existence means that stage has run.
    return {
        "fetch": modality(raw_dir(subject), subject, "T1"),
        "strip": modality(stripped_dir(subject), subject, "T1"),
        "segment": out_dir(subject) / "seg.nii.gz",
        "register": out_dir(subject) / "seg_in_mni.nii.gz",
    }.get(stage)
