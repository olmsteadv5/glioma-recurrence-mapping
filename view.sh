#!/usr/bin/env bash
# Open a segmented subject in ITK-SNAP with all four modalities loaded.
#
#   ./view.sh UPENN-GBM-00036_11            # prediction
#   ./view.sh UPENN-GBM-00002_11 reference  # expert labels instead
#
# In ITK-SNAP: the modality dropdown (top left) cycles T1GD/T1/T2/FLAIR with the
# segmentation staying overlaid, which is how you check a label against the
# evidence for it. "a" toggles the overlay on and off.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBJECT="${1:?usage: ./view.sh <SUBJECT> [reference]}"
WHICH="${2:-prediction}"
ALGO="${BRATS_ALGO:-BraTS25_2}"

SUBJ_DIR="$ROOT/data/PENN_GBM_nii/$SUBJECT"
[[ -d "$SUBJ_DIR" ]] || { echo "No such subject: $SUBJ_DIR" >&2; exit 1; }

if [[ "$WHICH" == "reference" ]]; then
  SEG="$ROOT/data/reference_seg/${SUBJECT}_seg.nii.gz"
else
  SEG="$ROOT/outputs/$ALGO/${SUBJECT}_seg.nii.gz"
fi
[[ -f "$SEG" ]] || { echo "No segmentation at: $SEG" >&2; exit 1; }

echo "main image : ${SUBJECT}_T1GD.nii.gz"
echo "overlays   : T1, T2, FLAIR"
echo "segmentation: $(basename "$SEG")  ($WHICH)"

exec itksnap \
  -g "$SUBJ_DIR/${SUBJECT}_T1GD.nii.gz" \
  -s "$SEG" \
  -l "$ROOT/brats_labels.itksnap.txt" \
  -o "$SUBJ_DIR/${SUBJECT}_T1.nii.gz" \
     "$SUBJ_DIR/${SUBJECT}_T2.nii.gz" \
     "$SUBJ_DIR/${SUBJECT}_FLAIR.nii.gz"
