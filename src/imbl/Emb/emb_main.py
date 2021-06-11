#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse
import json
import time
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.metrics import AUC

from save_model import output_files
from stat_mwb import under_samp


def parse_args():
    parser = argparse.ArgumentParser(description='Learned Embeddings script')
    parser.add_argument('--target', required=True, help='Target y variable in dataset')
    parser.add_argument('--period', default='Other', help='Delivery period (Pre, PI, Other)')
    parser.add_argument('--infile', required=True, help='Full path of input datafile')
    parser.add_argument('--samp_alg', default="RAND_und", help='"RAND_und" | "RAND_over" | None')
    parser.add_argument('--pred_alg', default="LearnEmb", help='Prediction method: LearnEmb | None')
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
    parser.add_argument('--feature_thresh', default=1.0,
                        help='How many of the corr_var_file variables used as features; [0.0, 1.0] is percentage; int > 1 is count')
    parser.add_argument('--batchsize', default=16, help='Size of batch')
    parser.add_argument('--epochs', default=20, help='Number of epochs to run')

    args = parser.parse_args()
    return args


# Given a filename containing sorted Cramer correlations and a threshold, return list of features
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


# load the dataset
def load_dataset(filename, cramer_coef, target, feature_thresh, row_count=None):
    # load the dataset as a pandas DataFrame
    if row_count:
        df = pd.read_csv(filename, header=0, index_col=0, nrows=row_count)
    else:
        df = pd.read_csv(filename, header=0, index_col=0)

    # split into input (X) and output (y) variables
    X = df.drop(target, axis=1, inplace=False)

    # order columns by Cramer coeffs
    cols = get_feature_list(cramer_coef, feature_thresh)
    X = X[cols]

    y = df[target]

    return X, y


# Given a DataFrame, return a similar one that is LabelEncded
def encode_df(df):
    df_enc = pd.DataFrame()
    for col in df:
        le = LabelEncoder()
        le.fit(df[col])
        df_enc[col] = le.transform(df[col])
    return df_enc


def main():
    opts = parse_args()
    print(f'opts = {opts}')
    BATCH_SIZE = int(opts.batchsize)
    EPOCHS = int(opts.epochs)
    feature_thr = int(opts.feature_thresh)

    # It is assumed that the input datafile already has collinear variables removed
    X, y = load_dataset(opts.infile, opts.corr_var_file, opts.target, feature_thr)

    print(f'\nIn main(): X.shape = {X.shape}; y.shape = {y.shape}\n')
    print(f'y.value_counts = {y.value_counts().values}')

    # Encode features using LabelEncoder
    X_enc_df = encode_df(X)

    if opts.seed:
        np.random.seed(int(opts.seed))

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_enc_df, y, stratify=y, test_size=0.30)

    #@todo - Move into mwb_common.sampler()
    sampler = RandomOverSampler(sampling_strategy=1.0)
    X_train, y_train= sampler.fit_resample(X_train, y_train)
    print(f'X_train.shape = {X_train.shape}; y_train.shape = {y_train.shape}')
    print(f'y_train.value_counts = {y_train.value_counts().values}')

    X_train_enc = np.array(X_train)
    X_test_enc = np.array(X_test)
    y_train_enc = np.array(y_train)
    y_test_enc = np.array(y_test)

    # make output 3d
    y_train_enc = y_train_enc.reshape((len(y_train_enc), 1, 1))
    y_test_enc = y_test_enc.reshape((len(y_test_enc), 1, 1))

    in_layers = list()
    em_layers = list()
    for col in range(X_train_enc.shape[1]):
        # calculate the number of unique inputs
        n_labels = len(np.unique(X_train_enc[:, col]))
        # define input layer
        in_layer = Input(shape=(1,))
        # define embedding layer
        em_layer = Embedding(n_labels+1, 10)(in_layer)  # MWB - Embedding docs say to use this
        # store layers
        in_layers.append(in_layer)
        em_layers.append(em_layer)

    # transpose input data to lists
    X_train_encl = []
    X_test_encl = []
    for col in range(X_train_enc.shape[1]):
        X_train_encl.append(X_train_enc[..., [col]])
        X_test_encl.append(X_test_enc[..., [col]])

    print(f'X_train_encl[0].shape = {X_train_encl[0].shape}')
    print(f'len(X_train_encl) = {len(X_train_encl)}')

    merge = concatenate(em_layers)
    dense = Dense(10, activation='relu', kernel_initializer='he_normal')(merge)
    output = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=in_layers, outputs=output)
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', sample_weight_mode='temporal',
                  metrics=['accuracy','Precision','Recall','AUC', AUC(name='pr_auc', curve='PR')])

    print(type(opts))
    run_start = time.time()
    model.fit(X_train_encl, y_train_enc, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    _, accuracy, prec, recall, auc, pr_auc = model.evaluate(X_test_encl, y_test_enc, verbose=0)
    predictions = model.predict(X_test_encl, batch_size=BATCH_SIZE)
    prob1 = predictions.flatten()
    y_pred = prob1 > 0.5

    # Create stratifed train_test split before under-sampling if opts.sample_tts is set
    output_files(y_test, y_pred, prob1, run_start, opts)


if __name__ == '__main__':
    main()