#
# This file contains functions for outputing
#
import time
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_recall_curve, \
    auc, precision_recall_fscore_support, matthews_corrcoef, accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, roc_curve, average_precision_score, fbeta_score
from imblearn.metrics import geometric_mean_score, sensitivity_specificity_support
from sklearn.dummy import DummyClassifier


def create_outfile_base(opts, params_dict=None):
    import datetime
    now = datetime.datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H%M%S"))
    if params_dict:
        param_str = "_".join([x if isinstance(x, str) else str(x) for x in params_dict.values()])
        fname = "-".join([opts.target, opts.period, opts.feats, opts.samp_alg, opts.pred_alg, param_str,\
                          str(opts.seed), str(opts.samp_strat), timestamp])
    else:
        fname = "-".join([opts.target, opts.period, opts.feats, opts.samp_alg, opts.pred_alg, str(opts.batchsize),\
                          str(opts.epochs), str(opts.seed), timestamp])
    fname = fname.replace(" ", "")
    return opts.output_dir + '/' + fname


def output_files(y_test, y_pred, prob1, run_start, opts):
    # Calculate stats based on algorithm results
    print(f'In save_to_file')
    print(f'np.bincount(y_test) =\n {np.bincount(y_test)}')
    print(f'np.bincount(y_pred) =\n {np.bincount(y_pred)}')
    accuracy_s = accuracy_score(y_test, y_pred)
    bal_accuracy_s = balanced_accuracy_score(y_test, y_pred)
    print(f'prob1.shape = {prob1.shape}')
    print(f'y_test.shape = {y_test.shape}')
    precision, recall, pr_thresh = precision_recall_curve(y_test, prob1)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_test, prob1)
    precision_s = precision_score(y_test, y_pred)
    recall_s = recall_score(y_test, y_pred)
    f1_s = f1_score(y_test, y_pred, average=None)
    fb1 = fbeta_score(y_test, y_pred, beta=1.0, average=None)
    fb1_m = fbeta_score(y_test, y_pred, beta=1.0, average='macro')
    fb2 = fbeta_score(y_test, y_pred, beta=2.0, average=None)
    fb2_m = fbeta_score(y_test, y_pred, beta=2.0, average='macro')
    fb05 = fbeta_score(y_test, y_pred, beta=0.5, average=None)
    fb05_m = fbeta_score(y_test, y_pred, beta=0.5, average='macro')
    precm, recm, f1m, suppm = precision_recall_fscore_support(y_test, y_pred, average="macro")
    fpr, tpr, thresholds = roc_curve(y_test, prob1)
    roc_auc = auc(fpr, tpr)
    roc_auc_prob = roc_auc_score(y_test, prob1)
    roc_auc_pred = roc_auc_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    gmeans = np.sqrt(tpr * (1 - fpr))
    g_ix = np.argmax(gmeans)
    gmean = geometric_mean_score(y_test, y_pred, average=None)
    gmean_ma = geometric_mean_score(y_test, y_pred, average='macro')
    sens, spec, ss_supp = sensitivity_specificity_support(y_test, y_pred, average=None)
    sens_m, spec_m, ss_supp_m = sensitivity_specificity_support(y_test, y_pred, average='macro')
    combStat = (precm + recm + f1m + mcc) / 4

    run_min = (time.time() - run_start) / 60
    outfile_base = create_outfile_base(opts)
    with open(outfile_base + '.out', 'w', newline='') as outfile:
        print(confusion_matrix(y_test, y_pred), file=outfile)
        print(f'\nClassification Report:\n {classification_report(y_test, y_pred)}', file=outfile)
        print(f'ROC_AUC = {roc_auc}', file=outfile)
        print(f'ROC_Prob = {roc_auc_prob}', file=outfile)
        print(f'ROC_Pred = {roc_auc_pred}', file=outfile)
        print(f'MCC = {mcc}', file=outfile)
        print(f'acc = {accuracy_s}', file=outfile)
        print(f'bal_acc = {bal_accuracy_s}', file=outfile)
        print(f'f1_score = {f1_s}', file=outfile)
        print(f'PR_AUC = {pr_auc}', file=outfile)
        print(f'Avg_precision = {avg_precision}', file=outfile)
        print(f'Combo = {combStat}', file=outfile)
        print(f'fb1 = {fb1}; fb1_macro = {fb1_m}', file=outfile)
        print(f'fb2 = {fb2}; fb2_macro = {fb2_m}', file=outfile)
        print(f'fb05 = {fb05}; fb2_macro = {fb05_m}', file=outfile)
        print(f'max_gmean = {gmeans[g_ix]}; max_thresh = {thresholds[g_ix]}', file=outfile)
        print(f'gmean = {gmean}; gmean_macro = {gmean_ma}', file=outfile)
        print(f'sens = {sens}; sens_macro = {sens_m}', file=outfile)
        print(f'spec = {spec}; spec_macro = {spec_m}', file=outfile)

    if opts.output_dir:
        import csv
        with open(outfile_base + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            writer.writerow(["run_time(min)", '{:.3f}'.format(run_min)])
            for arg in vars(opts):
                if arg in ["target","period","feats","samp_alg","pred_alg","batchsize","epochs","seed","samp_strat"]:
                    writer.writerow([arg, getattr(opts, arg)])

#            if opts.pred_params:
#                for key, value in params_dict.items():
#                    writer.writerow(["p_" + key, value])

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            ntn, nfp, nfn, ntp = confusion_matrix(y_test, y_pred, normalize='all').ravel()
            writer.writerow(["TN", tn])
            writer.writerow(["FP", fp])
            writer.writerow(["FN", fn])
            writer.writerow(["TP", tp])
            writer.writerow(["NTN", ntn])
            writer.writerow(["NFP", nfp])
            writer.writerow(["NFN", nfn])
            writer.writerow(["NTP", ntp])

            prec, rec, f1, supp = precision_recall_fscore_support(y_test, y_pred, average=None)
            writer.writerow(["acc", '{:.4f}'.format(accuracy_s)])
            writer.writerow(["bal_acc", '{:.4f}'.format(bal_accuracy_s)])
            writer.writerow(["precision_1", '{:.4f}'.format(prec[0])])
            writer.writerow(["recall_1", '{:.4f}'.format(rec[0])])
            writer.writerow(["F1_1", '{:.4f}'.format(f1[0])])
            writer.writerow(["precision_1", '{:.4f}'.format(prec[1])])
            writer.writerow(["recall_1", '{:.4f}'.format(rec[1])])
            writer.writerow(["F1_1", '{:.4f}'.format(f1[1])])

            writer.writerow(["precision_macro", '{:.4f}'.format(precm)])
            writer.writerow(["recall_macro", '{:.4f}'.format(recm)])
            writer.writerow(["F1_macro", '{:.4f}'.format(f1m)])

            writer.writerow(["ROC_AUC", '{:.4f}'.format(roc_auc)])
            writer.writerow(["ROC_Prob", '{:.4f}'.format(roc_auc_prob)])
            writer.writerow(["ROC_Pred", '{:.4f}'.format(roc_auc_pred)])

            writer.writerow(["PR_AUC", '{:.4f}'.format(pr_auc)])
            writer.writerow(["avg_precision", '{:.4f}'.format(avg_precision)])
            #writer.writerow(["naive_auc", '{:.4f}'.format(naive_auc)])
            writer.writerow(["MCC", '{:.4f}'.format(mcc)])

            # Create average meta-statistic for easy comparison (higher is better)
            writer.writerow(["Combo", '{:.4f}'.format(combStat)])

            # F-beta values
            writer.writerow(["Fb1_0", '{:.4f}'.format(fb1[0])])
            writer.writerow(["Fb1_1", '{:.4f}'.format(fb1[1])])
            writer.writerow(["Fb1_m", '{:.4f}'.format(fb1_m)])
            writer.writerow(["Fb2_0", '{:.4f}'.format(fb2[0])])
            writer.writerow(["Fb2_1", '{:.4f}'.format(fb2[1])])
            writer.writerow(["Fb2_m", '{:.4f}'.format(fb2_m)])
            writer.writerow(["Fb05_0", '{:.4f}'.format(fb05[0])])
            writer.writerow(["Fb05_1", '{:.4f}'.format(fb05[1])])
            writer.writerow(["Fb05_m", '{:.4f}'.format(fb05_m)])

            # Gmeans
            writer.writerow(["Gmean_0", '{:.4f}'.format(gmean[0])])
            writer.writerow(["Gmean_1", '{:.4f}'.format(gmean[1])])
            writer.writerow(["Gmean_ma", '{:.4f}'.format(gmean_ma)])
            writer.writerow(["Max Gmean", '{:.4f}'.format(gmeans[g_ix])])
            writer.writerow(["Max Thresh", '{:.4f}'.format(thresholds[g_ix])])

            # Sensitivity / Specificity
            writer.writerow(["Sens_0", '{:.4f}'.format(sens[0])])
            writer.writerow(["Sens_1", '{:.4f}'.format(sens[1])])
            writer.writerow(["Sens_ma", '{:.4f}'.format(sens_m)])
            writer.writerow(["Spec_0", '{:.4f}'.format(spec[0])])
            writer.writerow(["Spec_1", '{:.4f}'.format(spec[1])])
            writer.writerow(["Spec_ma", '{:.4f}'.format(spec_m)])

        # Output y_test / y_pred
        with open(outfile_base + '_pred.dat', 'w') as pred_file:
            np.savetxt(pred_file, (y_test, y_pred), delimiter=',', fmt='%1i')

        # Output predicted probs
        with open(outfile_base + '_probs.dat', 'w', newline='') as prob_file:
            np.savetxt(prob_file, prob1, delimiter=',')

        # Output fpr and tpr
        with open(outfile_base + '_fpr_tpr.dat', 'w', newline='') as ft_file:
            np.savetxt(ft_file, (fpr, tpr, thresholds), delimiter=',')

        # Output precision and recall
        if (len(pr_thresh) + 1) == len(precision):   # returned thresholds can be short by 1
            pr_thresh = np.append(pr_thresh, 1.0)
        with open(outfile_base + '_pr.dat', 'w', newline='') as pr_file:
            np.savetxt(pr_file, (precision, recall, pr_thresh), delimiter=',')
