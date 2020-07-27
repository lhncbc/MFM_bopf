#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse
import logging
import json
import time
import multiprocessing as mp

from sklearn.model_selection import train_test_split, ParameterGrid
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

from predict_out import save_to_file
from stat_mwb import under_samp


def parse_args():
    parser = argparse.ArgumentParser(description='MFM prediction script')
    parser.add_argument('--target', required=True, help='Target y variable in dataset')
    parser.add_argument('--infile', required=True, help='Full path of input datafile')
    parser.add_argument('--under_alg', default="random", help='"random" | <cohort varname>')
    parser.add_argument('--pred_alg', default="NB", help='Prediction method; NB | LR | SVC | LSVC | MLP | etc')
    parser.add_argument('--pred_params', default=None, help='String-based Dictionary of algorithm-specific tuning parameters')
    parser.add_argument('--seed', default=0, help='Initial random seed')
    parser.add_argument('--corr_var_file', required=True, help='Path of file containing list of highly-correlated variable names')
    parser.add_argument('--output_dir', default=None, help='Path to put outputs; if None, no output file produced')
    parser.add_argument('--samp_strat', default=1.0, help='Value of undersampling "sampling_strategy" param')
    parser.add_argument('--nproc', default=1, help='Number of proccesses to run')

    args = parser.parse_args()
    return args


def main():
    opts = parse_args()
    print(f'opts = {opts}')
    # It is assumed that the input datafile already has collinear variables removed
    df = pd.read_csv(opts.infile, index_col=0)
    X = df.drop(opts.target, axis=1, inplace=False)
    y = df[opts.target].values

    # Read list of Correlated variable names. No errors produced if the names don't match anything.
    corrVars = pd.read_csv(opts.corr_var_file, header=None)[0].to_list()
    X = X.loc[:, X.columns.intersection(corrVars)]
    print(f'X.columns = {X.columns}')

    if opts.seed:
        np.random.seed(int(opts.seed))

    # Create stratifed train_test split before under-sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30)
    print(f'type(X_train) = {type(X_train)}')
    print(f'type(y_train) = {type(y_train)}')
    print(f'X_train.head() = {X_train.head()}')
    print(f'y_train = {y_train}')

    # Start timer
    und_start = time.time()
    # Perform undersampling
    if opts.under_alg == 'random':
        X_res, y_res = under_samp(X_train, y_train, opts.samp_strat, opts.target, cohort=None)
        #rand_und = RandomUnderSampler(sampling_strategy = float(opts.samp_strat))
        #X_res, y_res = rand_und.fit_resample(X_train, y_train)
        print(f'X_res =\n {X_res}; y_res=\n{y_res}')
        print(f'np.bincount(y_res)={np.bincount(y_res)}')
    else:
        X_res, y_res = cohort_under(opts.under_alg)
    und_end = time.time()

    params_dict = json.loads(opts.pred_params)
    params_list = list(ParameterGrid(params_dict))
    print(f'\n\nparams_list = {params_list}\n\n')

    pool = mp.Pool(processes=int(opts.nproc))
    for params in params_list:
        if int(opts.nproc) > 1:
            pool.apply_async(clf_predict, args=(params, X_res, y_res, y_test, opts, X_test))
        else:
            clf_predict(params, X_res, y_res, y_test, opts, X_test)

    pool.close()
    pool.join()


def clf_predict(params, X_res, y_res, y_test, opts, X_test):
    try:
        clf = None
        if opts.pred_alg == 'NB':
            clf = GaussianNB(**params)
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

        # Classifier
        clf_start = time.time()
        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_test)
        save_to_file(y_test, y_pred, X_test, clf, clf_start, opts, params)
    except Exception as e:
        print(f'caught exception in worker thread: {e}')
        raise e

if __name__ == '__main__':
    main()