#!/bin/bash
#SBATCH --time=0-00:40:00
#SBATCH --mem-per-cpu=4000
#SBATCH --array=0-7
#SBATCH --output=./slurm_logs/slurm-%A_%a.out

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

SUBJECT=$1

ml purge
module load teflon
ml anaconda3
source activate mne

cd "$INPUT_DIR" || exit 1

CHUNKSIZE=15
n=$SLURM_ARRAY_TASK_ID

rm "$OUTPUT_DIR"/no_*_matches.txt 2> /dev/null

if [ -n "$SUBJECT" ]
then
  filenames=("$SUBJECT"_*_tsss_mc.fif)
  indexes=0
else
  filenames=(*_tsss_mc.fif)
  indexes=`seq $((n*CHUNKSIZE)) $(((n + 1)*CHUNKSIZE - 1))`
fi

for i in $indexes
do
  f=${filenames[$i]}
  sub=${f%%_*}
  task=${f#*_}
  task=${task%%_*}
  if [ -n "$sub" ]
  then
    echo "$sub"
    echo "$task"
    srun xvfb-run -a python /scratch/nbe/tbi-meg/veera/pipeline/ica/run_ica.py "${sub}" "${task}" "${INPUT_DIR}"/"${f}" "${OUTPUT_DIR}"
  fi
done