#
# This file contains functions for outputing
#
import time
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_recall_curve, \
    auc, precision_recall_fscore_support, matthews_corrcoef


def create_outfile_base(opts, params_dict=None):
    import datetime
    now = datetime.datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H%M%S"))
    if params_dict:
        param_str = "_".join([x if isinstance(x, str) else str(x) for x in params_dict.values()])
        fname = "-".join([opts.target, opts.sample_tts, opts.under_alg, opts.pred_alg, param_str, str(opts.seed), str(opts.samp_strat), timestamp])
    else:
        fname = "-".join([opts.target, opts.sample_tts, opts.under_alg, opts.pred_alg, str(opts.seed), timestamp])
    fname = fname.replace(" ", "")
    return opts.output_dir + '/' + fname


def save_to_file(X_train, y_train, X_test, y_test, y_pred, clf, clf_start, opts, params_dict):

    # Calculate stats based on algorithm results
    print(f'In save_to_file')
    probs = clf.predict_proba(X_test)
    probs = probs[:, 1]  # Only positives
    precision, recall, _ = precision_recall_curve(y_test, probs, pos_label=2)
    pr_auc = auc(recall, precision)
    precm, recm, f1m, suppm = precision_recall_fscore_support(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    combStat = (precm + recm + f1m + mcc) / 4

    clf_min = (time.time() - clf_start) / 60
    with open(create_outfile_base(opts, params_dict) + '.out', 'w', newline='') as outfile:
        print(f'X_train.shape =\n {X_train.shape}; y_train.shape=\n{y_train.shape}')
        print(f'X_test.shape =\n {X_test.shape}; y_test.shape=\n{y_test.shape}')
        print(f'np.bincount(y_train)={np.bincount(y_train)}')
        print(f'np.bincount(y_test)={np.bincount(y_test)}')

        print(f'\n\nclf.get_params() = {clf.get_params()}\n\n', file=outfile)
        print(confusion_matrix(y_test, y_pred), file=outfile)
        print(f'\nClassification Report:\n {classification_report(y_test, y_pred)}', file=outfile)
        print(f'ROC_AUC = {roc_auc}', file=outfile)
        print(f'MCC = {mcc}', file=outfile)
        print(f'f1_score = {f1_score(y_test, y_pred, average=None)}', file=outfile)
        print(f'PR_AUC = {pr_auc}', file=outfile)
        print(f'Combo = {combStat}', file=outfile)

        if opts.pred_alg == 'LR':
            coeffs = pd.Series(data=np.abs(clf.coef_[0]), index=X_test.columns.values)
            coeffs.sort_values(inplace=True,ascending=False)
            print(f'coeffs = \n{coeffs}', file=outfile)

    if opts.output_dir:
        import csv
        with open(create_outfile_base(opts, params_dict) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            writer.writerow(["CLF_time(min)", '{:.3f}'.format(clf_min)])
            for arg in vars(opts):
                if arg in ["target", "under_alg", "pred_alg", "seed", "samp_strat"]:
                    writer.writerow([arg, getattr(opts, arg)])

            if opts.pred_params:
                for key, value in params_dict.items():
                    writer.writerow(["p_" + key, value])

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            writer.writerow(["TN", tn])
            writer.writerow(["FP", fp])
            writer.writerow(["FN", fn])
            writer.writerow(["TP", tp])

            prec, rec, f1, supp = precision_recall_fscore_support(y_test, y_pred, average=None)
            writer.writerow(["precision_1", '{:.4f}'.format(prec[0])])
            writer.writerow(["recall_1", '{:.4f}'.format(rec[0])])
            writer.writerow(["F1_1", '{:.4f}'.format(f1[0])])
            writer.writerow(["precision_2", '{:.4f}'.format(prec[1])])
            writer.writerow(["recall_2", '{:.4f}'.format(rec[1])])
            writer.writerow(["F1_2", '{:.4f}'.format(f1[1])])

            writer.writerow(["precision_macro", '{:.4f}'.format(precm)])
            writer.writerow(["recall_macro", '{:.4f}'.format(recm)])
            writer.writerow(["F1_macro", '{:.4f}'.format(f1m)])

            writer.writerow(["ROC_AUC", '{:.4f}'.format(roc_auc)])

            writer.writerow(["PR_AUC", '{:.4f}'.format(pr_auc)])
            writer.writerow(["MCC", '{:.4f}'.format(mcc)])

            # Create average meta-statistic for easy comparison (higher is better)
            writer.writerow(["Combo", '{:.4f}'.format(combStat)])