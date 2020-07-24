import pandas as pd
import numpy as np
import argparse
import logging
import time
import random
import multiprocessing as mp

from stat_mwb import under_samp
from stat_mwb import cramers_v_df
from stat_mwb import theils_u_df

# Global variable is rather ugly. @TODO - figure out way to remove global
results = []


def parse_args():
    parser = argparse.ArgumentParser(description='Undersampling comparison script')
    parser.add_argument('--runs', default=1, help='Number of random runs to average')
    parser.add_argument('--infile', required=True, help='Full path of input datafile')
    parser.add_argument('--target', required=True, help='Target y variable in dataset')
    parser.add_argument('--sampling_strat', default=1.0, help='Sampling strategy; if float ratio of minority/majority')
    parser.add_argument('--under', default='RAND', help='NONE: no undersampling; RAND: Random Undersampling; otherwise expected to be variable for cohort undersampling')
    parser.add_argument('--corr_alg', default='Cramer', help='Correlation algorithm: Cramer or Theil')
    parser.add_argument('--seed', help='Initial random seed')
    parser.add_argument('--outdir', default='./', help='output data file to be used')
    parser.add_argument('--no_output', help='Do not save results to file')
    parser.add_argument('--nprocs', default=1, help='How many processes to fork')

    args = parser.parse_args()
    return args


def under_corr(X, y, opts):
    if opts.under == 'NONE':
        X_res = X
        y_res = y
    else:
        X_res, y_res = under_samp(X, y, float(opts.sampling_strat), opts.target, opts.under)

    all_df = X_res.copy()
    all_df.insert(0, opts.target, y_res)

    # @TODO - only calculate "target" column of correlation.
    #  In case of Theil, make sure it is the correct direction
    if opts.corr_alg == 'Cramer':
        corr_und = cramers_v_df(all_df)
    elif opts.corr_alg == 'Theil':
        corr_und = theils_u_df(all_df)
    else:
        raise ValueError("Invalid correlation alg: {opts.corr_alg}; must be either 'Cramer' or 'Theil'")

    # Replace any NaNs with 0
    corr_und.fillna(0, inplace=True)
    return corr_und


# Ugly use of global. @TODO - remove global
def collect_results(result):
    results.append(result)
    #print(f'len(results) = {len(results)}')


# Need to reset random seed for each process
def proc_init():
    np.random.seed()


def main():
    logging.basicConfig(filename='ucm.log', level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%y%m%d-%H:%M:%S')
    opts = parse_args()
    print(f'opts = {opts}')

    if opts.seed:
        print(f'Setting np.random.seed = {opts.seed}')
        np.random.seed(int(opts.seed))

    # It is assumed that the input datafile already has collinear variables removed
    df = pd.read_csv(opts.infile, index_col=0)
    X = df.drop(opts.target, axis=1, inplace=False)
    y = df[opts.target]

    pool = mp.Pool(processes=int(opts.nprocs), initializer=proc_init)
    for run in range(0, int(opts.runs)):
        pool.apply_async(under_corr, args=(X, y, opts), callback=collect_results)

    pool.close()
    pool.join()

    # Combine results from different runs and average them
    rank_df = pd.DataFrame()
    corr_df = pd.DataFrame()
    for run, corr_und_df in enumerate(results):
#        print(f'type(corr_df) = {type(corr_und_df)}')
#        print(f'corr_df.head() = {corr_und_df.head(10)}')
        targ_corr = corr_und_df[opts.target].sort_values(ascending=False)
        targ_corr.name = "corr" + str(run)
        targ_rank = targ_corr.rank(method='average', ascending=False)
        targ_rank.name = "rank" + str(run)
        if len(rank_df) == 0:
            corr_df = pd.DataFrame(targ_corr)
            rank_df = pd.DataFrame(targ_rank)
        else:
            c1_df = pd.DataFrame(targ_corr)
            corr_df = pd.merge(corr_df, c1_df, left_index=True, right_index=True)
            r1_df = pd.DataFrame(targ_rank)
            rank_df = pd.merge(rank_df, r1_df, left_index=True, right_index=True)

    # Add columns for the average correlation and rank, and sort by them
    corr_df['mean_corr'] = corr_df.apply(np.average, axis=1)
    corr_df.sort_values(by='mean_corr', ascending=False, inplace=True)
    print(corr_df.head(20))
    rank_df['mean_rank'] = rank_df.apply(np.average, axis=1)
    rank_df.sort_values(by='mean_rank', inplace=True)
    print(rank_df.head(20))

    if not opts.no_output:
        import datetime
        pd.options.display.float_format = '${:,.5f}'.format
        now = datetime.datetime.now()
        timestamp = str(now.strftime("%Y%m%d_%H%M%S"))

        corrfile = opts.outdir + '/corr_' + opts.target + '_' + str(opts.under) + '_' + \
                   opts.corr_alg + '_' + opts.seed + '_' + timestamp + '.csv'
        corr_df.to_csv(corrfile, header=True, float_format='%.6f')

        rankfile = opts.outdir + '/rank_' + opts.target + '_' + str(opts.under) + '_' + \
                   opts.corr_alg + '_' + opts.seed + '_' + timestamp + '.csv'
        rank_df.to_csv(rankfile, header=True)


if __name__ == '__main__':
    main()
