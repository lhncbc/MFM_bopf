import pandas as pd
import numpy as np
import argparse
import logging
import os

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from stat_mwb import cramers_v_df


def parse_args():
    parser = argparse.ArgumentParser(description='Undersampling comparison script')
    parser.add_argument('--runs', default=1, help='Number of random runs to average')
    parser.add_argument('--infile', required=True, help='Full path of input datafile')
    parser.add_argument('--target', required=True, help='Target y variable in dataset')
    parser.add_argument('--sampling_strat', default=1.0, help='Target y variable in dataset')
    parser.add_argument('--seed', help='Initial random seed')
    parser.add_argument('--outdir', help='output data file to be used')
    parser.add_argument('--no_output', help='Do not save results to file')

    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(filename='ucm.log', level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%y%m%d-%H:%M:%S')
    opts = parse_args()

    # It is assumed that the input datafile already has collinear variables removed
    df = pd.read_csv(opts.infile, index_col=0)
    X = df.drop(opts.target, axis=1, inplace=False)
    y = df[opts.target].values

    if opts.seed:
        np.random.seed(int(opts.seed))

    rank_df = pd.DataFrame()
    # Should consider multi-processing this loop
    for run in range(0, int(opts.runs)):
        rand_und = RandomUnderSampler(sampling_strategy=opts.sampling_strat)
        X_res, y_res = rand_und.fit_resample(X, y)
        y_res_s = pd.Series(y_res)
        xind = X.columns
        X_res_df = pd.DataFrame(X_res, columns=xind)
        print(Counter(y_res))

        # Not sure if this is necessary, but in Jupyter Notebook values are floats and
        # we need ints.
        for col in X_res_df:
            X_res_df[col] = X_res_df[col].astype(int)

        y_res_s.name = opts.target
        all_df = pd.DataFrame(X_res_df)
        all_df.insert(0, opts.target, y_res_s)

        # Could use this block of code to eliminate NaNs in Cramer, but then some
        # variables wouldn't be included
        for col in all_df:
            if all_df[col].unique().shape[0] == 1:
                print(col)

        c_und = cramers_v_df(all_df)
        c_und.fillna(0, inplace=True)
        targ_corr = c_und[opts.target].sort_values(ascending=False)
        targ_corr.name = "corr" + str(run)
        targ_rank = targ_corr.rank(method='min', ascending=False).astype(int)
        targ_rank.name = "rank" + str(run)
        if len(rank_df) == 0:
            corr_df = pd.DataFrame(targ_corr)
            rank_df = pd.DataFrame(targ_rank)
        else:
            c1_df = pd.DataFrame(targ_corr)
            corr_df = pd.merge(corr_df, c1_df, left_index=True, right_index=True)
            r1_df = pd.DataFrame(targ_rank)
            rank_df = pd.merge(rank_df, r1_df, left_index=True, right_index=True)

    corr_df['mean_corr'] = corr_df.apply(np.average, axis=1)
    corr_df.sort_values(by='mean_corr', ascending=False, inplace=True)
    print(corr_df.head(20))
    rank_df['mean_rank'] = rank_df.apply(np.average, axis=1)
    rank_df.sort_values(by='mean_rank', inplace=True)
    print(rank_df.head(20))


    if not opts.no_output:
        import datetime
        now = datetime.datetime.now()
        timestamp = str(now.strftime("%Y%m%d_%H%M%S"))
        #corrfile = '../../data/csl/corr_' + timestamp + '.csv'
        corrfile = opts.outdir + '/corr_' + timestamp + '.csv'
        if not os.path.exists(corrfile):
            corr_df.to_csv(corrfile, header=True)

        #rankfile = '../../data/csl/rank_' + timestamp + '.csv'
        rankfile = opts.outdir + '/rank_' + timestamp + '.csv'
        if not os.path.exists(rankfile):
            rank_df.to_csv(rankfile, header=True)

if __name__ == '__main__':
    main()