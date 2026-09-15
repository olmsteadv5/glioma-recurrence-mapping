# Run the BraTS adult-glioma segmenter in Docker.

import shutil
import subprocess

from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms as Algorithms

from paths import modality, out_dir

# BraTS25_1 placed first but is CUDA-only; BraTS25_2 is the best CPU-capable entry
GPU_ALGORITHM = Algorithms.BraTS25_1
CPU_ALGORITHM = Algorithms.BraTS25_2


def cuda_available():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def run(subject, stripped):
    # Segment one subject; ~3 min on CPU, faster on a GPU host.
    gpu = cuda_available()
    algorithm = GPU_ALGORITHM if gpu else CPU_ALGORITHM
    if not gpu:
        print(f"    no CUDA device; using {algorithm.value} (2nd place) instead of "
              f"{GPU_ALGORITHM.value}")

    destination = out_dir(subject) / "seg.nii.gz"
    segmenter = AdultGliomaPreAndPostTreatmentSegmenter(algorithm=algorithm,
                                                        force_cpu=not gpu)
    segmenter.infer_single(
        t1c=modality(stripped, subject, "T1GD"),
        t1n=modality(stripped, subject, "T1"),
        t2f=modality(stripped, subject, "FLAIR"),
        t2w=modality(stripped, subject, "T2"),
        output_file=destination,
        log_file=out_dir(subject) / "segmentation.log",
    )
    if not destination.exists():
        raise RuntimeError(f"{subject}: segmenter wrote no output")
    return destination
