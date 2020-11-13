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

ml purge
module load teflon
ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

python /scratch/nbe/tbi-meg/veera/pipeline/ica/run_ica_corrmap.py 1 "${AR_TYPE}"