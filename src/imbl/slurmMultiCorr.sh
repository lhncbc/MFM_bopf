#!/bin/bash

#SBATCH --workdir=.
#SBATCH --output=slurmOut/slurm_%j.out
#SBATCH --error=slurmOut/slurm_%j.error
#SBATCH --job-name=multiCor
#SBATCH --cpus-per-task=10
#SBATCH --partition=VM

# Dynamically create outputs directory
#mkdir outputs_$SLURM_JOB_ID

env | grep SLURM
which python
python --version
time python multiCorrelate.py --target transfus_yes --infile ../../data/csl/CSL_d5_transfus_yes_2020-07.csv --seed 0 --outdir ./underCompOut --under high_Age --corr_alg Cramer --runs 20 --nprocs 10
