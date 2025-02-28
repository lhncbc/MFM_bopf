# Runs on lhc-lx-mbopf
time python ../predict_main.py --target transfus_yes --period='PI' --infile ../../../data/csl/CSL_ty_PI.csv --pred_alg RF --corr_var_file ../../../data/csl/CramerTheil/Cramer_PI_Ty_coeff_Union50.csv --feats='Union50' --under_alg='NONE' --samp_strat 0.5 --sample_tts 0 --feature_thresh 1.0 --sample_weights=False --pred_params '[{"n_estimators":[128],"criterion":["gini"],"max_depth":[40],"min_samples_leaf":[1],"min_samples_split":[2],"max_leaf_nodes":[100],"max_features":["auto"],"class_weight":["balanced"],"random_state":[7]}]' --nproc 1 --seed 3 --output_dir ./output

