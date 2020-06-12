#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse
import logging
import json
import time

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, classification_report, roc_auc_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc


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

    args = parser.parse_args()
    return args


def create_outfile(opts, params_dict=None):
    import datetime
    now = datetime.datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H%M%S"))
    if params_dict:
        param_str = "_".join([x if isinstance(x, str) else str(x) for x in params_dict.values()])
        fname = "-".join([opts.target, opts.under_alg, opts.pred_alg, param_str, str(opts.seed), timestamp])
    else:
        fname = "-".join([opts.target, opts.under_alg, opts.pred_alg, str(opts.seed), timestamp])
    return opts.output_dir + '/' + fname + '.csv'


def main():
    logging.basicConfig(filename='mpred.log', level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%y%m%d-%H:%M:%S')
    opts = parse_args()
    print(f'opts = {opts}')
    if opts.pred_params:
        params_dict=json.loads(opts.pred_params)
    else:
        params_dict=None

    # It is assumed that the input datafile already has collinear variables removed
    df = pd.read_csv(opts.infile, index_col=0)
    X = df.drop(opts.target, axis=1, inplace=False)
    y = df[opts.target].values

    # Read list of Correlated variable names. No errors produced if the names don't match anything.
    corrVars = pd.read_csv(opts.corr_var_file, header=None)[0].to_list()
    X = X.loc[:, X.columns.intersection(corrVars)]

    if opts.seed:
        np.random.seed(int(opts.seed))


    # Create stratifed train_test split before under-sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30)

    # Start timer
    und_start = time.time()

    # Perform undersampling
    if opts.under_alg == 'random':
        rand_und = RandomUnderSampler(sampling_strategy=1.0)
        X_res, y_res = rand_und.fit_resample(X_train, y_train)
    else:
        X_res, y_res = cohort_under(opts.under_alg)

    und_end = time.time()

    # Make prediction based on input algorithm
    if opts.pred_alg == 'NB':
        clf = GaussianNB()
    elif opts.pred_alg == 'LR':
        clf = LogisticRegression(**params_dict)
    elif opts.pred_alg == 'RF':
        clf = RandomForestClassifier(**params_dict)
    elif opts.pred_alg == 'SVC':
        clf = SVC(**params_dict)
    elif opts.pred_alg == 'LSVC':
        lsvm = LinearSVC(**params_dict)
        # predict_proba() not available for LinearSVC; use CalibratedClassifierCV
        clf = CalibratedClassifierCV(lsvm)

    # Classifier
    clf_start = time.time()
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test)

    print(f'\n\nclf.get_params() = {clf.get_params()}\n\n')
    clf_min = (time.time() - clf_start)/60

    from sklearn.metrics import matthews_corrcoef
    print(confusion_matrix(y_test, y_pred))
#    print(f'Accuracy = {accuracy_score(y_test, y_pred)}')
#    print(f'Recall = {recall_score(y_test, y_pred)}')
    print(f'\nClassification Report:\n {classification_report(y_test, y_pred)}')
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f'ROC_AUC = {roc_auc}')
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f'MCC = {mcc}')

    print(f'f1_score = {f1_score(y_test, y_pred, average=None)}')
#    print(f'balanced_accuracy = {balanced_accuracy_score(y_test, y_pred)}')
#    print(f'ROC_AUC_each = {roc_auc_score(y_test, y_pred, average=None)}')


    probs = clf.predict_proba(X_test)
    probs = probs[:, 1]  # Only positives
    precision, recall, _ = precision_recall_curve(y_test, probs, pos_label=2)
    pr_auc = auc(recall, precision)
    precm, recm, f1m, suppm = precision_recall_fscore_support(y_test, y_pred, average="macro")

    # Removing pr_auc from combo stat since it isn't acting consistently
    #if opts.pred_alg == 'SVC':
    combStat = (precm + recm + f1m + mcc) / 4
    #else:
    #    combStat = (precm + recm + f1m + pr_auc + mcc) / 5

    print(f'PR_AUC = {pr_auc}')
    print(f'Combo = {combStat}')

    if opts.output_dir:
        print(f'\n\noutfile = {create_outfile(opts, params_dict)}\n\n')
        import csv
        with open(create_outfile(opts, params_dict), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            writer.writerow(["CLF_time(min)", '{:.3f}'.format(clf_min)])
            for arg in vars(opts):
                if arg in ["target", "under_alg", "pred_alg", "seed"]:
                    writer.writerow([arg, getattr(opts, arg)])

            if opts.pred_params:
                for key, value in params_dict.items():
                    writer.writerow(["p_"+key, value])

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            writer.writerow(["TN", tn])
            writer.writerow(["FP", fp])
            writer.writerow(["FN", fn])
            writer.writerow(["TP", tp])

            prec, rec, f1, supp = precision_recall_fscore_support(y_test, y_pred, average=None)
            writer.writerow(["precision_2", '{:.3f}'.format(prec[1])])
            writer.writerow(["recall_2", '{:.3f}'.format(rec[1])])
            writer.writerow(["F1_2", '{:.3f}'.format(f1[1])])

            writer.writerow(["precision_macro", '{:.3f}'.format(precm)])
            writer.writerow(["recall_macro", '{:.3f}'.format(recm)])
            writer.writerow(["F1_macro", '{:.3f}'.format(f1m)])

            writer.writerow(["ROC_AUC", '{:.3f}'.format(roc_auc)])

            writer.writerow(["PR_AUC", '{:.3f}'.format(pr_auc)])
            writer.writerow(["MCC", '{:.3f}'.format(mcc)])

            # Create average meta-statistic for easy comparison (higher is better)
            writer.writerow(["Combo", '{:.3f}'.format(combStat)])


    '''
        corrfile = '../../data/csl/corr_' + timestamp + '.csv'
        if not os.path.exists(corrfile):
            corr_df.to_csv(corrfile, header=True)

        rankfile = '../../data/csl/rank_' + timestamp + '.csv'
        if not os.path.exists(rankfile):
            rank_df.to_csv(rankfile, header=True)
    '''
if __name__ == '__main__':
    main()