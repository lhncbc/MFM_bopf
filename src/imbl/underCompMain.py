import pandas as pd
import numpy as np
import argparse
import logging
import os

from stat_mwb import under_samp
from stat_mwb import cramers_v_df
from stat_mwb import theils_u_df


def parse_args():
    parser = argparse.ArgumentParser(description='Undersampling comparison script')
    parser.add_argument('--runs', default=1, help='Number of random runs to average')
    parser.add_argument('--infile', required=True, help='Full path of input datafile')
    parser.add_argument('--target', required=True, help='Target y variable in dataset')
    parser.add_argument('--sampling_strat', default=1.0, help='Sampling strategy; if float ratio of minority/majority')
    parser.add_argument('--cohort', default=None, help='If not "None", the variable to be used for cohort undersampling')
    parser.add_argument('--corr_alg', default='Cramer', help='Correlation algorithm: Cramer or Theil')
    parser.add_argument('--seed', help='Initial random seed')
    parser.add_argument('--outdir', default='./', help='output data file to be used')
    parser.add_argument('--no_output', help='Do not save results to file')

    args = parser.parse_args()

    return args


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

    rank_df = pd.DataFrame()
    # Should consider multi-processing this loop
    for run in range(0, int(opts.runs)):
        X_res, y_res = under_samp(X, y, float(opts.sampling_strat), opts.target, opts.cohort)
        #print(f'y_res = {y_res}')
        #print(f'X_res = \n{X_res}')

        all_df = X_res.copy()
        all_df.insert(0, opts.target, y_res)
        #print(f'X_res = \n{X_res}')
        #print(f'all_df = \n{all_df}')

        # Could use this block of code to eliminate NaNs in Cramer/Theil, but then some
        # variables wouldn't be included in the summary DataFrame
        for col in all_df:
            if all_df[col].unique().shape[0] == 1:
                print(col)

        if opts.corr_alg == 'Cramer':
            corr_und = cramers_v_df(all_df)
        elif opts.corr_alg == 'Theil':
            corr_und = theils_u_df(all_df)
        else:
            raise ValueError("Invalid correlation alg: {opts.corr_alg}; must be either 'Cramer' or 'Theil'")

        corr_und.fillna(0, inplace=True)
        targ_corr = corr_und[opts.target].sort_values(ascending=False)
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

        corrfile = opts.outdir + '/corr_' + opts.target + '_' + str(opts.cohort) + '_' + \
                   opts.corr_alg + '_' + opts.seed + '_' + timestamp + '.csv'
        if not os.path.exists(corrfile):
            corr_df.to_csv(corrfile, header=True)

        rankfile = opts.outdir + '/rank_' + opts.target + '_' + str(opts.cohort) + '_' + \
                   opts.corr_alg + '_' + opts.seed + '_' + timestamp + '.csv'
        if not os.path.exists(rankfile):
            rank_df.to_csv(rankfile, header=True)

        corr_und_file = opts.outdir + '/corr_und_' + opts.target + '_' + str(opts.cohort) + '_' + \
                        opts.corr_alg + '_' + opts.seed + '_' + timestamp + '.csv'
        if not os.path.exists(corr_und_file):
            corr_und.to_csv(corr_und_file, header=True)


if __name__ == '__main__':
    main()
