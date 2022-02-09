#!/bin/bash

#SBATCH --workdir=.
#SBATCH --output=slurmOut/slurm_%j.out
#SBATCH --error=slurmOut/slurm_%j.error
#SBATCH --job-name=multPred
#SBATCH --cpus-per-task=15
#SBATCH --partition=VM

# Dynamically create outputs directory
#mkdir outputs_$SLURM_JOB_ID

env 
echo $PATH
which python
python --version
time python predict_main.py --target transfus_yes --infile ../../data/csl/CSL_d5_transfus_yes_2020-08-21.csv --pred_alg LR --corr_var_file ../../data/csl/top50-transfus_yes-Cramer.txt --samp_strat 1.0 --pred_params '[{"random_state":[7], "solver":["liblinear"] ,"max_iter":[10000, 100000, 1000000], "C":[100, 1000]}]' --nproc 4 --seed 7 --output_dir ./output
