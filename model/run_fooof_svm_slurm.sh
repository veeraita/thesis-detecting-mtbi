#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --mem-per-cpu=3000
#SBATCH --output=./slurm_logs/slurm-%A.out

ml purge
module load teflon
ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

srun xvfb-run -a python -u run_fooof_svm.py -n
