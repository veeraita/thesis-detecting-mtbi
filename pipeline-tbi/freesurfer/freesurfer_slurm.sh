#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem-per-cpu=8000    # 8000MB of memory
#SBATCH --array=0-47
#SBATCH --output=./slurm_logs/slurm-%A_%a.out

if  [ -z "$MRI_DIR" ]
then
  echo "MRI_DIR not set, exiting"
  exit 1
fi

if  [ -z "$SUBJECTS_DIR" ]
then
  echo "SUBJECTS_DIR not set, exiting"
  exit 1
fi

echo "MRI_DIR set as $MRI_DIR"
echo "SUBJECTS_DIR set as $SUBJECTS_DIR"
temp=$SUBJECTS_DIR

ml purge
module load teflon
module load freesurfer

export SUBJECTS_DIR=$temp

cd $MRI_DIR || exit 1

dirnames=(*/)
sub_dir=${dirnames[$SLURM_ARRAY_TASK_ID]}
sub=${sub_dir%?}
if [ -n "$sub" ]
then
  printf -v sub "%03d" "$sub"
  files=("$sub_dir"/*.dcm)
  file="${files[0]}"
  if [ -n "$file" ]
    then
      echo "${sub}"
      echo "${file}"
      srun recon-all -s "${sub}" -i "${file}" -all
  fi
fi

