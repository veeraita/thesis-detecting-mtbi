#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --mem-per-cpu=4000
#SBATCH --array=0-10
#SBATCH --output=./slurm_logs/slurm-%A_%a.out

if  [ -z "$INPUT_DIR" ]
then
  echo "INPUT_DIR not set, exiting"
  exit 1
fi

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

echo "INPUT_DIR set as $INPUT_DIR"
echo "SUBJECTS_DIR set as $SUBJECTS_DIR"
echo "OUTPUT_DIR set as $OUTPUT_DIR"

ml purge
module load teflon
ml anaconda3
source activate mne


cd $SUBJECTS_DIR

CHUNKSIZE=5
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
    srun python /scratch/nbe/tbi-meg/veera/pipeline/psd/get_psd.py ${sub} ${INPUT_DIR} ${SUBJECTS_DIR} ${OUTPUT_DIR}
  fi
done

