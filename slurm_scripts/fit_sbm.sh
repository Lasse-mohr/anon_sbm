#!/bin/sh
### General options
#BSUB -q gpua100
#BSUB -J anon_sbm_fit
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 20GB
#BSUB -W 12:00
#BSUB -o slurm_outputs/anon_sbm_fit_%J.out
#BSUB -e slurm_outputs/anon_sbm_fit_%J.err

module purge
module load python3/3.12.11

source /zhome/c1/2/109045/anon_sbm/bin/activate

python3 src/pipelines/run_all.py

