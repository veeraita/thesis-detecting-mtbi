#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --mem-per-cpu=8000
#SBATCH --array=0-5
#SBATCH --output=./slurm_logs/slurm-%A_%a.out

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

echo "SUBJECTS_DIR set as $SUBJECTS_DIR"
echo "OUTPUT_DIR set as $OUTPUT_DIR"

ml anaconda3
source activate mne

cd $SUBJECTS_DIR || exit 1

CHUNKSIZE=10
n=$SLURM_ARRAY_TASK_ID
indexes=`seq $((n*CHUNKSIZE)) $(((n + 1)*CHUNKSIZE - 1))`

dirnames=(*/)
for i in $indexes
do
  sub=${dirnames[$i]}
  sub=${sub%?}
  if [ -n "$sub" ]
  then
    echo "${sub}"
    srun xvfb-run -a python /scratch/nbe/tbi-meg/veera/pipeline/trans/calc_trans.py ${sub} ${SUBJECTS_DIR} ${OUTPUT_DIR}
  fi
done
