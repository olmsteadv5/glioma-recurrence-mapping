# Run the BraTS adult-glioma segmenter in Docker.

import shutil
import subprocess

from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms as Algorithms

from paths import modality, out_dir

# BraTS25_1 won the challenge but needs CUDA; BraTS25_2 is the best CPU one
GPU_ALGORITHM = Algorithms.BraTS25_1
CPU_ALGORITHM = Algorithms.BraTS25_2


def cuda_available():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def stop():
    # brats detaches its container, so killing python leaves inference running
    listed = subprocess.run(["docker", "ps", "--format", "{{.ID}} {{.Image}}"],
                            capture_output=True, text=True)
    running = [line.split()[0] for line in listed.stdout.splitlines()
               if line.split()[1].startswith("brainles/")]
    if running:
        subprocess.run(["docker", "stop", *running], capture_output=True)
    return len(running)


def run(subject, stripped):
    # About 3 minutes per subject on CPU.
    gpu = cuda_available()
    algorithm = GPU_ALGORITHM if gpu else CPU_ALGORITHM
    if not gpu:
        print(f"  no GPU, using {algorithm.value}")

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
    return destination, algorithm.value, "gpu" if gpu else "cpu"
