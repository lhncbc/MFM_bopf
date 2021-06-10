# Single run for balanced class_weight
time python emb_main.py --target trans_loss --period='PI' --infile ../../../data/csl/CSL_tl_PI.csv --pred_alg Emb --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='RAND_und' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 16 --epochs 2 --output_dir ./output

time python emb_main.py --target trans_loss --period='Pre' --infile ../../../data/csl/CSL_tl_Pre.csv --pred_alg Emb --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Tl_coeff_Union50.csv --feats='Union50' --samp_alg='RAND_und' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 16 --epochs 2 --output_dir ./output

time python emb_main.py --target transfus_yes --period='PI' --infile ../../../data/csl/CSL_ty_PI.csv --pred_alg Emb --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='RAND_und' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 16 --epochs 2 --output_dir ./output

time python emb_main.py --target transfus_yes --period='Pre' --infile ../../../data/csl/CSL_ty_Pre.csv --pred_alg Emb --corr_var_file ../../../data/csl/CramerTheil/Cramer_Pre_Ty_coeff_Union50.csv --feats='Union50' --samp_alg='RAND_und' --samp_strat 1.0 --feature_thresh 100 --nproc 1 --seed 3 --batchsize 16 --epochs 2 --output_dir ./output

