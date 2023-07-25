#!/bin/bash

#SBATCH --job-name=1gpu_runner
#SBATCH --output=../../../results/%j_1gpu_hpc_runner_slurm_log.txt
#SBATCH --account=bgmp                                                      ### specific to UO HPC
#SBATCH --partition=a100gpu                                                 ### specific to UO HPC
#SBATCH --nodes=1
#SBATCH --gpus=1

mkdir ../../../results/$SLURM_JOB_ID

conda activate spike_env

/usr/bin/time -v python hpc_runner.py
