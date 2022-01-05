# Single run for balanced class_weight
time python imbl_TF.py --target trans_loss --period='PI' --infile ../../../data/csl/CSL_tl_PI.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 256 --epochs 100 --output_dir ./output

time python imbl_TF.py --target trans_loss --period='Pre' --infile ../../../data/csl/CSL_tl_Pre.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 256 --epochs 100 --output_dir ./output

time python imbl_TF.py --target transfus_yes --period='PI' --infile ../../../data/csl/CSL_ty_PI.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 256 --epochs 100 --output_dir ./output

time python imbl_TF.py --target transfus_yes --period='Pre' --infile ../../../data/csl/CSL_ty_Pre.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 256 --epochs 100 --output_dir ./output

#### Batchsize 2048
time python imbl_TF.py --target trans_loss --period='PI' --infile ../../../data/csl/CSL_tl_PI.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 2048 --epochs 100 --output_dir ./output

time python imbl_TF.py --target trans_loss --period='Pre' --infile ../../../data/csl/CSL_tl_Pre.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 2048 --epochs 100 --output_dir ./output

time python imbl_TF.py --target transfus_yes --period='PI' --infile ../../../data/csl/CSL_ty_PI.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 2048 --epochs 100 --output_dir ./output

time python imbl_TF.py --target transfus_yes --period='Pre' --infile ../../../data/csl/CSL_ty_Pre.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 2048 --epochs 100 --output_dir ./output

#### Batchsize 64
time python imbl_TF.py --target trans_loss --period='PI' --infile ../../../data/csl/CSL_tl_PI.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 64 --epochs 100 --output_dir ./output

time python imbl_TF.py --target trans_loss --period='Pre' --infile ../../../data/csl/CSL_tl_Pre.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 64 --epochs 100 --output_dir ./output

time python imbl_TF.py --target transfus_yes --period='PI' --infile ../../../data/csl/CSL_ty_PI.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 64 --epochs 100 --output_dir ./output

time python imbl_TF.py --target transfus_yes --period='Pre' --infile ../../../data/csl/CSL_ty_Pre.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 64 --epochs 100 --output_dir ./output


#### Batchsize 512
time python imbl_TF.py --target trans_loss --period='PI' --infile ../../../data/csl/CSL_tl_PI.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 512 --epochs 100 --output_dir ./output

time python imbl_TF.py --target trans_loss --period='Pre' --infile ../../../data/csl/CSL_tl_Pre.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 512 --epochs 100 --output_dir ./output

time python imbl_TF.py --target transfus_yes --period='PI' --infile ../../../data/csl/CSL_ty_PI.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 512 --epochs 100 --output_dir ./output

time python imbl_TF.py --target transfus_yes --period='Pre' --infile ../../../data/csl/CSL_ty_Pre.csv --pred_alg TFIM --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='ClassWeight' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 512 --epochs 100 --output_dir ./output

