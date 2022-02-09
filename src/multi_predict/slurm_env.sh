#!/bin/bash

#SBATCH --chdir=.
#SBATCH --output=slurmOut/slurm_%j.out
#SBATCH --error=slurmOut/slurm_%j.error
#SBATCH --job-name=envTest
#SBATCH --cpus-per-task=1
#SBATCH --partition=VM

env
