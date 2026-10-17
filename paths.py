
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
REFERENCE_SEG = INPUTS / "reference_seg"
ATLAS_INPUT = INPUTS / "atlas"
ATLAS_DIR = OUTPUTS / "atlas"
RESULTS = OUTPUTS / "results.json"
SCANS = OUTPUTS / "scans"
# explainer: per patient, anything that needs both scans
PAIRS = OUTPUTS / "pairs"
ANALYSIS = OUTPUTS / "results"
PARTICIPANTS = INPUTS / "participants.tsv"

MNI = Path("/home/vilem/Documents/SeizureAnalysis/from_downloads/MNI152_T1_1mm_Brain.nii.gz")
SYNTHSTRIP = Path("/opt/freesurfer/bin/mri_synthstrip")

MODALITIES = ("T1", "T1GD", "T2", "FLAIR")
LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
REGIONS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3), "RC": (4,)}

THREADS = "12"


# decision: PREDICT-GBM ids start at 50000 so they never clash with UPenn's 1-630
COLLECTIONS = {5: "PREDGBM-T-", 6: "PREDGBM-L-", 7: "PREDGBM-R-"}
PREDICT = INPUTS / "predict_gbm"
RHUH = INPUTS / "rhuh_gbm"


def subject_id(number, timepoint=1):
    return f"{COLLECTIONS.get(number // 10000, 'UPENN-GBM-')}{number:05d}_{timepoint}1"


def external(number):
    return number // 10000 in COLLECTIONS


def anatomy(directory, subject):
    # decision: fall back to T1-Gd, PREDICT-GBM ships no T1
    t1 = modality(directory, subject, "T1")
    return t1 if t1.exists() else modality(directory, subject, "T1GD")


def raw_dir(subject):
    return INPUTS / subject


def out_dir(subject):
    return SCANS / subject


def cache_dir(subject):
    # explainer: viewer cache, safe to delete
    path = out_dir(subject) / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pair_dir(subject):
    path = PAIRS / subject[10:15]
    path.mkdir(parents=True, exist_ok=True)
    return path


def pair_cache(subject):
    path = pair_dir(subject) / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def walk_dir(subject):
    path = pair_dir(subject) / "walkdown"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(space, kind="tables"):
    # decision: split by space so native and mni never overwrite each other
    path = ANALYSIS / space / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def stripped_dir(subject):
    return out_dir(subject) / "stripped"


def modality(directory, subject, name):
    return directory / f"{subject}_{name}.nii.gz"


STAGES = ("fetch", "strip", "tissue", "segment", "register", "align", "map")


def stage_file(subject, stage):
    return {
        "fetch": anatomy(raw_dir(subject), subject),
        "strip": anatomy(stripped_dir(subject), subject),
        "tissue": out_dir(subject) / "tissue.nii.gz",
        "segment": out_dir(subject) / "seg.nii.gz",
        "register": out_dir(subject) / "seg_in_mni.nii.gz",
        # explainer: pre-op scans have nothing to align, patient space is the follow-up's
        "align": (out_dir(subject) / "seg.nii.gz" if subject.endswith("_11")
                  else pair_dir(subject) / "first_seg_in_later.nii.gz"),
    }.get(stage)
