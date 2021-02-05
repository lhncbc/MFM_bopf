#!/bin/bash

#SBATCH --chdir=.
#SBATCH --output=slurmOut/slurm_%j.out
#SBATCH --error=slurmOut/slurm_%j.error
#SBATCH --job-name=multPred
#SBATCH --cpus-per-task=15
#SBATCH --partition=VM

env 
echo $PATH
which python
python --version

#export optional environment variables as defaults
if [[ $SLURM_EXPORT_ENV != *"sampling_strat"* ]];then
    sampling_strat=1.0
fi
if [[ $SLURM_EXPORT_ENV != *"output_dir"* ]];then
    output_dir=./output
fi

time python ../predict_main.py --target $target --infile $infile --under_alg $under_alg --pred_alg $pred_alg --corr_var_file $corr_var_file --pred_params $pred_params --samp_strat $samp_strat --corr_var_file $corr_var_file --nproc $nproc --seed $seed --sample_tts $sample_tts --feature_thresh feature_thresh --output_dir $output_dir
