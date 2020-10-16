#!/bin/bash

if  [ -z "$SUBJECTS_DIR" ]
then
  echo "SUBJECTS_DIR not set, exiting"
  exit 1
fi

if  [ -z "$OUTPUT_DIR" ]
then
  echo "OUTPUT_DIR not set, exiting"
  exit 1
fi

AVERAGES_DIR=/scratch/nbe/tbi-meg/veera/averages/normative

echo "SUBJECTS_DIR set as $SUBJECTS_DIR"
echo "OUTPUT_DIR set as $OUTPUT_DIR"

ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

cd $SUBJECTS_DIR || exit 1

dirnames=(*/)

for d in "${dirnames[@]}"
do
  sub=${d%?}
  task=EC
  if [ -n "$sub" ]
  then
    echo "${sub}"
    python /scratch/nbe/tbi-meg/veera/pipeline/tmap/calc_tmap.py $sub $task $OUTPUT_DIR $SUBJECTS_DIR $AVERAGES_DIR
  fi
done
