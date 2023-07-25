#!/bin/bash

#SBATCH --job-name=make_db
#SBATCH --output=./logs/make_db-logs_%j.txt
#SBATCH --account=bgmp                                                      ### specific to UO HPC
#SBATCH --partition=bgmp                                                    ### specific to UO HPC
#SBATCH --nodes=1

conda activate spike_env

/usr/bin/time -v python make_db.py