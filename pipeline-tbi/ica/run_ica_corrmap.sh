#!/bin/bash

if  [ -z "$INPUT_DIR" ]
then
  echo "INPUT_DIR not set, exiting"
  exit 1
fi

if  [ -z "$OUTPUT_DIR" ]
then
  echo "OUTPUT_DIR not set, exiting"
  exit 1
fi

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