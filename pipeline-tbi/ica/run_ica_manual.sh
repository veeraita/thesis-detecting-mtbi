#!/bin/bash

INPUT_DIR=/scratch/work/italinv1/tbi/meg
OUTPUT_DIR=/scratch/nbe/tbi-meg/veera/processed

echo "INPUT_DIR set as $INPUT_DIR"
echo "OUTPUT_DIR set as $OUTPUT_DIR"

AR_TYPE=$1

if  [ -z "$AR_TYPE" ]
then
  echo "Artifact type argument [ecg,eog] is required"
  exit 1
fi

FNAME=no_${AR_TYPE}_matches.txt

ml purge
module load teflon
ml anaconda3
source activate mne

python /scratch/nbe/tbi-meg/veera/pipeline/ica/run_ica_manual.py --file "${OUTPUT_DIR}"/"${FNAME}"