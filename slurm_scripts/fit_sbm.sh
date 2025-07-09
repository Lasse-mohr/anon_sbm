#!/bin/sh
### General options
#BSUB -q hpc
#BSUB -J anon_sbm_fit
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -M 20GB
#BSUB -W 1:00
#BSUB -o slurm_outputs/anon_sbm_fit_%J.out
#BSUB -e slurm_outputs/anon_sbm_fit_%J.err

module purge
module load python3/3.12

source /zhome/c1/2/109045/anon_sbm_env/bin/activate

python3 src/pipelines/fit_sbm.py --fit_config /zhome/c1/2/109045/anon_sbm/configs/sbm_fit_block_size_experiments.yml

