#!/usr/bin/env python
from distutils.util import strtobool

import pandas as pd
import numpy as np
import argparse
import logging
import json
import time
import multiprocessing as mp

from sklearn.model_selection import train_test_split, ParameterGrid
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

from predict_out import save_to_file
from stat_mwb import under_samp


def parse_args():
    parser = argparse.ArgumentParser(description='MFM prediction script')
    parser.add_argument('--target', required=True, help='Target y variable in dataset')
    parser.add_argument('--period', default='Other', help='Delivery period (Pre, PI, Other)')
    parser.add_argument('--infile', required=True, help='Full path of input datafile')
    parser.add_argument('--under_alg', default="RAND", help='"RAND" | <cohort varname>')
    parser.add_argument('--pred_alg', default="NB",
                        help='Prediction method; NB | LR | SVC | LSVC | MLP | etc')
    parser.add_argument('--pred_params', default=None,
                        help='String-based Dictionary of algorithm-specific tuning parameters')
    parser.add_argument('--seed', default=0, help='Initial random seed')
    # Following two params are a bit redundant
    # @todo - Create cor_var_file name based on feat, period, and target
    parser.add_argument('--feats', default='ALL', help='Feature file used(ALL, top50, Union50)')
    parser.add_argument('--corr_var_file', required=True,
                        help='Path of file containing list of highly-correlated variable names')
    parser.add_argument('--output_dir', default=None,
                        help='Path to put outputs; if None, no output file produced')
    parser.add_argument('--samp_strat', default=1.0,
                        help='Value of undersampling "sampling_strategy" param')
    parser.add_argument('--nproc', default=1, help='Number of proccesses to run')
    parser.add_argument('--sample_tts', default=0,
                        help='Sample both the test and train data; if not set, only sample training data')
    parser.add_argument('--feature_thresh', default=1.0,
                        help='How many of the corr_var_file variables used as features; [0.0, 1.0] is percentage; int > 1 is count')
    parser.add_argument('--sample_weights', dest='sample_weights', default=False,
                        type=lambda x: bool(strtobool(x)),
                        help='Are sample_weights to be used; currently only applies to GB')

    args = parser.parse_args()
    return args


# Given a filename containing sorted Cramer correlations and a threshold, return list of features
# @todo - Reconsider how this might interact with the "feature_file" param. (ie, do Union50 & feature_thresh conflict?)
def get_feature_list(filename, feature_thresh):
    corr_var_df = pd.read_csv(filename, header=None, sep='\t', index_col=0, names=['Variable', 'corr'])
    corr_var_list = corr_var_df.index.to_list()
    if feature_thresh > 1:  # Assuming integer count
        thresh = int(min(feature_thresh, len(corr_var_list)))
    else:  # Assuming float percentage
        thresh = int(feature_thresh * len(corr_var_list))

    print(f'len(cor_var_list = {len(corr_var_list)}')
    print(f'thresh = {thresh}')
    return corr_var_list[:thresh]


# Calculate the sample weights based on the training set for y
# Currently only applicable to GradientBoosting
def calc_sample_weights(y_train):
    from sklearn.utils import class_weight
    weights = class_weight.compute_sample_weight(class_weight="balanced", y=y_train)
    print(np.unique(weights))
    print(f'y_train.value_counts(): {y_train.value_counts()}')
    print(f'len(y_train): {len(y_train)}')
    return weights


def main():
    opts = parse_args()
    print(f'opts = {opts}')
    # It is assumed that the input datafile already has collinear variables removed
    df = pd.read_csv(opts.infile, index_col=0)
    X = df.drop(opts.target, axis=1, inplace=False)
    y = df[opts.target].values

    print(f'\nIn main(): X.shape = {X.shape}; y.shape = {y.shape}\n')

    # Read list of Correlated variable names. No errors produced if the names don't match anything.
    #corrVars = pd.read_csv(opts.corr_var_file, header=None)[0].to_list()

    # Get list of highly correlated variables (features) limited by the threshold
    corrVars = get_feature_list(opts.corr_var_file, float(opts.feature_thresh))
    if opts.under_alg not in corrVars:
        if opts.under_alg != 'NONE' and opts.under_alg != 'RAND':
            corrVars.append(opts.under_alg)

    print(f'\nIn main()2: X.shape = {X.shape}; y.shape = {y.shape}\n')

    X = X.loc[:, X.columns.intersection(corrVars)]

    if opts.seed:
        np.random.seed(int(opts.seed))

    # Create stratifed train_test split before under-sampling if opts.sample_tts is set
    if int(opts.sample_tts) == 1:
        print('sample_tts == 1')
        #@TODO: refactor this code to use different cross-validation strategies.
        print('Sampling test data changes the data distribution and results are usually considered INVALID')
        import warnings
        warnings.warn("Should never sub-sample test data")
        # No undersampling - NOTE: this combination makes no sense - can't sample tts if NONE
        if opts.under_alg == 'NONE':
            X_res_t, y_res_t = X, y
        else:
            X_res_t, y_res_t = under_samp(X, y, float(opts.samp_strat), opts.target, opts.under_alg)

        X_res, X_test, y_res, y_test = train_test_split(X_res_t, y_res_t, stratify=y_res_t, test_size=0.30)
    else:
        print('sample_tts != 1')
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30)
        # No undersampling - NOTE: this could take considerably longer in prediction
        if opts.under_alg == 'NONE':
            X_res, y_res = X_train, y_train
        else:
            X_res, y_res = under_samp(X_train, y_train, float(opts.samp_strat), opts.target, opts.under_alg)

    params_dict = json.loads(opts.pred_params)
    params_list = list(ParameterGrid(params_dict))
    print(f'\n\nparams_list = {params_list}\n\n')

    pool = mp.Pool(processes=int(opts.nproc))
    for params in params_list:
        if int(opts.nproc) > 1:
            pool.apply_async(clf_predict, args=(params, X_res, y_res, X_test, y_test, opts))
        else:
            clf_predict(params, X_res, y_res, X_test, y_test, opts)

    pool.close()
    pool.join()


def clf_predict(params, X_train, y_train, X_test, y_test, opts):
    try:
        print(f'In clf_predict')
        print(f'params= {params}')
        # Convert any "None" value strings into the Python None value
        for k,v in params.items():
            if v == "None":
                params[k] = None

        clf = None
        if opts.pred_alg == 'NB':
            clf = GaussianNB(**params)
        # WARNING: CategoricalNB not currently working due to bug that may be fixed in
        #          sklearn 0.24.
        elif opts.pred_alg == 'CNB':
            clf = CategoricalNB(**params)
        elif opts.pred_alg == 'LR':
            clf = LogisticRegression(**params)
        elif opts.pred_alg == 'RF':
            clf = RandomForestClassifier(**params)
        elif opts.pred_alg == 'SVC':
            clf = SVC(**params)
        elif opts.pred_alg == 'LSVC':
            lsvm = LinearSVC(**params)
            # predict_proba() not available for LinearSVC; use CalibratedClassifierCV
            clf = CalibratedClassifierCV(lsvm)
        elif opts.pred_alg == 'MLP':
            clf = MLPClassifier(**params)
        elif opts.pred_alg == 'GB':
            clf = GradientBoostingClassifier(**params)

        # Classifier
        clf_start = time.time()
        if opts.sample_weights:
            assert opts.pred_alg == 'GB', "Error: only GB allowed with sample_weights"
            weights = calc_sample_weights(y_train)
            clf.fit(X_train, y_train, sample_weight=weights)
        else:
            clf.fit(X_train, y_train)

        print('After clf.fit')
        y_pred = clf.predict(X_test)
        save_to_file(X_train, y_train, X_test, y_test, y_pred, clf, clf_start, opts, params)
        if hasattr(clf, 'n_iter_'):
            print(f'clf.n_iter_ = {clf.n_iter_}')
    except Exception as e:
        print(f'caught exception in worker thread: {e}')
        raise e


if __name__ == '__main__':
    main()