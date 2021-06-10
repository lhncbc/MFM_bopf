#!/bin/bash

#SBATCH --output=slurmOut/slurm_%j.out
#SBATCH --error=slurmOut/slurm_%j.error
#SBATCH --job-name=multiCor
#SBATCH --cpus-per-task=10
#SBATCH --partition=VM

# Dynamically create outputs directory
#mkdir outputs_$SLURM_JOB_ID

env
echo $PATH
which python
python --version

#export environment variables as defaults
if [[ $SLURM_EXPORT_ENV != *"sampling_strat"* ]];then
    sampling_strat=1.0
fi
if [[ $SLURM_EXPORT_ENV != *"outdir"* ]];then
    outdir=./underCompOut
fi

time python multiCorrelate.py --target $target --infile $infile --seed $seed --under $under --sampling_strat $sampling_strat --corr_alg $corrAlg --runs $runs --nprocs $nprocs --outdir $outdir
