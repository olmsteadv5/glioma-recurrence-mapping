import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paths import THREADS, out_dir, stripped_dir

# explainer: TumorSynth's nnU-Net environment and weights, installed outside the repo
HOME = Path.home() / ".local" / "share" / "tumorsynth"
MODELS = HOME / "nnUNet_v1.7" / "nnUNet_trained_models"
PREDICT = HOME / "env" / "bin" / "nnUNet_predict"
# decision: T1-Gd first as the authors recommend, then the other three, each predicted alone and then voted
SEQUENCES = ("T1GD", "T1", "T2", "FLAIR")
TISSUE_LABELS = 18


def _predict(image, task, work):
    # explainer: the same call mri_TumorSynth makes, one sequence per run
    folder, result = work / f"in_{image.stem}", work / f"out_{image.stem}"
    folder.mkdir()
    shutil.copy(image, folder / "input_0000.nii.gz")
    env = {**os.environ, "RESULTS_FOLDER": str(MODELS), "nnUNet_raw_data_base": str(work / "raw"),
           "nnUNet_preprocessed": str(work / "pre"), "OMP_NUM_THREADS": THREADS, "MKL_NUM_THREADS": THREADS}
    subprocess.run([PREDICT, "-i", folder, "-o", result, "-tr", "nnUNetTrainerV2", "-ctr", "nnUNetTrainerV2CascadeFullRes",
                    "-m", "3d_fullres", "-p", "nnUNetPlansv2.1", "-t", task, "-f", "0", "1", "2", "3", "4"],
                   env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # explainer: read into memory now, the temporary folder is gone once the run finishes
    image = nib.load(next(result.glob("*.nii.gz")))
    return np.asarray(image.dataobj).astype(np.int16), image.affine, image.header


def run(subject):
    dest = out_dir(subject) / "tumorsynth.nii.gz"
    if dest.exists():
        return dest
    images = [stripped_dir(subject) / f"{subject}_{s}.nii.gz" for s in SEQUENCES]
    images = [i for i in images if i.exists()]
    with tempfile.TemporaryDirectory(prefix="tumorsynth_") as tmp:
        work = Path(tmp)
        outputs = [_predict(i, "Task002_Tumor", work) for i in images]
    # explainer: their fusion, the share of runs voting for each label, then the label with the largest share
    votes = np.zeros((TISSUE_LABELS + 1,) + outputs[0][0].shape, np.float32)
    for labels, _, _ in outputs:
        for k in range(1, TISSUE_LABELS + 1):
            votes[k] += labels == k
    fused = np.argmax(votes, axis=0).astype(np.int16)
    nib.save(nib.Nifti1Image(fused, outputs[0][1], outputs[0][2]), dest)
    return dest


if __name__ == "__main__":
    for subject in sys.argv[1:]:
        print(subject, run(subject), flush=True)
