#
# This file contains functions for outputing
#
import time

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_recall_curve, \
    auc, precision_recall_fscore_support, matthews_corrcoef


def create_outfile_base(opts, params_dict=None):
    import datetime
    now = datetime.datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H%M%S"))
    if params_dict:
        param_str = "_".join([x if isinstance(x, str) else str(x) for x in params_dict.values()])
        fname = "-".join([opts.target, opts.under_alg, opts.pred_alg, param_str, str(opts.seed), timestamp])
    else:
        fname = "-".join([opts.target, opts.under_alg, opts.pred_alg, str(opts.seed), timestamp])

    return opts.output_dir + '/' + fname


def save_to_file(y_test, y_pred, X_test, clf, clf_start, opts, params_dict):

    # Calculate stats based on algorithm results
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
        print(f'\n\nclf.get_params() = {clf.get_params()}\n\n', file=outfile)
        print(confusion_matrix(y_test, y_pred), file=outfile)
        print(f'\nClassification Report:\n {classification_report(y_test, y_pred)}', file=outfile)
        print(f'ROC_AUC = {roc_auc}', file=outfile)
        print(f'MCC = {mcc}', file=outfile)
        print(f'f1_score = {f1_score(y_test, y_pred, average=None)}', file=outfile)
        print(f'PR_AUC = {pr_auc}', file=outfile)
        print(f'Combo = {combStat}', file=outfile)
    if opts.output_dir:
        import csv
        with open(create_outfile_base(opts, params_dict) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            writer.writerow(["CLF_time(min)", '{:.3f}'.format(clf_min)])
            for arg in vars(opts):
                if arg in ["target", "under_alg", "pred_alg", "seed"]:
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