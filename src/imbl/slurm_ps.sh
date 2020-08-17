#!/bin/bash

#SBATCH --workdir=.
#SBATCH --output=slurmOut/slurm_%j.out
#SBATCH --error=slurmOut/slurm_%j.error
#SBATCH --job-name=psVM
#SBATCH --cpus-per-task=1
#SBATCH --partition=VM

# Dynamically create outputs directory
#mkdir outputs_$SLURM_JOB_ID

#ps -ef | grep python
hostname -a
ps -ef| grep python 
printf "\n*****sleep 5\n\n"; sleep 5
ps -ef | grep python
printf "\n*****sleep 5\n\n"; sleep 5
ps -ef | grep python
printf "\n*****sleep 5\n\n"; sleep 5
ps -ef | grep python
printf "\n*****sleep 5\n\n"; sleep 5
ps -ef | grep python
printf "\n*****sleep 5\n\n"; sleep 5
ps -ef | grep python
printf "\n*****sleep 5\n\n"; sleep 5
ps -ef | grep python
printf "\n*****sleep 5\n\n"; sleep 5
ps -ef | grep python
