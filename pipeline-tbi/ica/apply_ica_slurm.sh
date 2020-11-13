#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --mem-per-cpu=4000
#SBATCH --output=./slurm_logs/slurm-%A.out

INPUT_DIR=/scratch/work/italinv1/tbi/meg
OUTPUT_DIR=/scratch/nbe/tbi-meg/veera/processed

echo "INPUT_DIR set as $INPUT_DIR"
echo "OUTPUT_DIR set as $OUTPUT_DIR"

SUBJECT=$1

ml purge
module load teflon
ml anaconda3
source activate mne

cd "$INPUT_DIR" || exit 1

if [ -n "$SUBJECT" ]
then
  filenames=("$SUBJECT"_*_tsss_mc.fif)
else
  filenames=(*_tsss_mc.fif)
fi

for f in "${filenames[@]}"
do
  sub=${f%%_*}
  task=${f#*_}
  task=${task%%_*}
  echo "$sub"
  echo "$task"
  python /scratch/nbe/tbi-meg/veera/pipeline/ica/apply_ica.py "${sub}" "${task}" "${INPUT_DIR}"/"${f}" "${OUTPUT_DIR}"
done