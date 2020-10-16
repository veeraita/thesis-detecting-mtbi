#!/bin/bash

if  [ -z "$OUTPUT_DIR" ]
then
  echo "OUTPUT_DIR not set, exiting"
  exit 1
fi

echo "OUTPUT_DIR set as $OUTPUT_DIR"

ml purge
module load teflon
ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne


cd $OUTPUT_DIR


dirnames=(*/)
for d in "${dirnames[@]}"
do
  sub=${d%?}
  if [ -n "$sub" ]
  then
    echo "${sub}"
    python /scratch/nbe/tbi-meg/veera/pipeline/rpsd/relative_psd.py ${sub} ${OUTPUT_DIR}
  fi
done

