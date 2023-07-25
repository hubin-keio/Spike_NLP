#!/bin/bash

#SBATCH --job-name=1gpu_runner
#SBATCH --output=../../../results/%j_1gpu_runner_slurm_log.txt
#SBATCH --account=bgmp                                                      ### specific to UO HPC
#SBATCH --partition=a100gpu                                                 ### specific to UO HPC
#SBATCH --nodes=1
#SBATCH --gpus=1

conda activate spike_env

/usr/bin/time -v python runner.py
