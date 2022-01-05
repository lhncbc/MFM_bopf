#!/usr/bin/env python
# coding: utf-8

# #### Copyright 2019 The TensorFlow Authors.
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


# # Classification on imbalanced data
# ## Modified by Mike Bopf for CSL data

# ## Setup
import time

import tensorflow as tf
from tensorflow import keras

import argparse
import os
from collections import Counter
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from data_util import get_feature_list, load_dataset, encode_df
from save_model import output_files


def parse_args():
    parser = argparse.ArgumentParser(description='Learned Embeddings script')
    parser.add_argument('--target', required=True, help='Target y variable in dataset')
    parser.add_argument('--period', default='Other', help='Delivery period (Pre, PI, Other)')
    parser.add_argument('--infile', required=True, help='Full path of input datafile')
    parser.add_argument('--samp_alg', default="RAND_ovr", help='"RAND_ovr" | "ClassWeight"')
    parser.add_argument('--pred_alg', default="TFIM", help='Prediction method: TFIM | None')
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

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def make_model(input_dim, metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=input_dim),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model


def main():
    opts = parse_args()
    print(f'opts = {opts}')
    BATCH_SIZE = int(opts.batchsize)
    EPOCHS = int(opts.epochs)
    feature_thr = int(opts.feature_thresh)

    # It is assumed that the input datafile already has collinear variables removed
    #X, y = load_dataset(opts.infile, opts.corr_var_file, opts.target,
    #                    feature_thr, row_count=2000)
    X, y = load_dataset(opts.infile, opts.corr_var_file, opts.target,
                        feature_thr)

    print(f'\nIn main(): X.shape = {X.shape}; y.shape = {y.shape}\n')
    print(f'y.value_counts = {y.value_counts().values}')

    neg, pos = np.bincount(y)
    total = neg + pos
    print('Examples:\nTotal: {}\nPositive: {} ({:.2f}% of total)\n'.format(
           total, pos, 100 * pos / total))

    if opts.seed:
        np.random.seed(int(opts.seed))

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=1)
    X_train_f = np.array(X_train)
    X_val_f = np.array(X_val)
    X_test_f = np.array(X_test)
    y_train_l = np.array(y_train)
    y_val_l = np.array(y_val)
    y_test_l = np.array(y_test)
    print(f'X_train_f.shape: {X_train_f.shape}; y_train.shape: {y_train.shape}')
    print(f'X_val_f.shape: {X_val_f.shape}; y_val.shape: {y_val.shape}')

    input_shape = (X_train_f.shape[-1], )
    print(f'input_shape: {input_shape}')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=2,
        patience=10,
        mode='max',
        restore_best_weights=True)

    initial_bias = np.log([pos / neg])
    print(f'initial_bias: {initial_bias}')

    model = make_model(input_shape, output_bias=initial_bias)

    run_start = 0
    # Case where Class Weighting is used to handle data imbalance
    if opts.samp_alg == 'ClassWeight':

        # Scaling by total/2 helps keep the loss to a similar magnitude.
        # The sum of the weights of all examples stays the same.
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))

        # Train a model with class weights
        # Note: Using `class_weights` changes the range of the loss. This may affect the stability of the training depending on the optimizer. Optimizers whose step size is dependent on the magnitude of the gradient, like `optimizers.SGD`, may fail. The optimizer used here, `optimizers.Adam`, is unaffected by the scaling change. Also note that because of the weighting, the total losses are not comparable between the two models.
        #model.load_weights(initial_weights)

        run_start = time.time()
        model_history = model.fit(X_train_f, y_train_l, verbose=2,
            batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping],
            validation_data=(X_val_f, y_val_l), class_weight=class_weight)

    # Oversampling
    elif opts.samp_alg == 'RAND_ovr':
        # Handle over-sampling case
        pass


    # MWB - make commmon between both types of imbalance handling
    train_predictions = model.predict(X_train_f, batch_size=BATCH_SIZE)
    test_predictions = model.predict(X_test_f, batch_size=BATCH_SIZE)
    prob1 = test_predictions.flatten()
    y_pred = prob1 > 0.5

    from collections import Counter
    print(f'Counter(y_pred) = {Counter(y_pred)}')

    output_files(y_test_l, y_pred, prob1, run_start, opts)

    #weighted_results = weighted_model.evaluate(X_test_f, y_test_l,
    #                                       batch_size=BATCH_SIZE, verbose=2)
#for name, value in zip(weighted_model.metrics_names, weighted_results):
#    print(name, ': ', value)

#zip_iter = zip(weighted_model.metrics_names, weighted_results)
#metric_dict = dict(zip_iter)
#p = metric_dict['precision']
#r = metric_dict['recall']
#f2 = fbeta2(p, r)
#print(f'f2 = {f2}')

#pos_features = X_train_f[y_train_l == 1]
#neg_features = X_train_f[y_train_l == 0]
#
#pos_labels = y_train_l[y_train_l == 1]
#neg_labels = y_train_l[y_train_l == 0]

#from collections import Counter
#print(type(y_train_l))
#print(Counter(y_train_l))
#print(len(pos_features))
#print(len(neg_features))
#print(len(pos_labels))
#print(len(neg_labels))

# #### Using NumPy
# 
# You can balance the dataset manually by choosing the right number of random 
# indices from the positive examples:

# In[47]:

"""
ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

res_pos_features.shape

# In[48]:


resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

resampled_features.shape


# #### Using `tf.data`

# If you're using `tf.data` the easiest way to produce balanced examples is to start with a `positive` and a `negative` dataset, and merge them. See [the tf.data guide](../../guide/data.ipynb) for more examples.

# In[49]:


def make_ds(features, labels):
    BUFFER_SIZE = 100000
    ds = tf.data.Dataset.from_tensor_slices((features, labels))  # .cache()
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds


pos_ds = make_ds(pos_features, pos_labels)
neg_ds = make_ds(neg_features, neg_labels)

for features, label in pos_ds.take(1):
    print("Features:\n", features.numpy())
    print()
    print("Label: ", label.numpy())

resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

for features, label in resampled_ds.take(1):
    print(label.numpy().mean())

resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)
resampled_steps_per_epoch

resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

val_ds = tf.data.Dataset.from_tensor_slices((X_val_f, y_val_l)).cache()
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

resampled_history = resampled_model.fit(
    resampled_ds,
    epochs=EPOCHS,
    steps_per_epoch=resampled_steps_per_epoch,
    callbacks=[early_stopping],
    validation_data=val_ds)

resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

resampled_history = resampled_model.fit(
    resampled_ds,
    # These are not real epochs
    steps_per_epoch=20,
    epochs=10 * EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_ds))

train_predictions_resampled = resampled_model.predict(X_train_f, batch_size=BATCH_SIZE)
test_predictions_resampled = resampled_model.predict(X_test_f, batch_size=BATCH_SIZE)

resampled_results = resampled_model.evaluate(X_test_f, y_test_l,
                                             batch_size=BATCH_SIZE, verbose=2)
for name, value in zip(resampled_model.metrics_names, resampled_results):
    print(name, ': ', value)

zip_iter = zip(resampled_model.metrics_names, resampled_results)
metric_dict = dict(zip_iter)
p = metric_dict['precision']
r = metric_dict['recall']
f2 = fbeta2(p, r)
print(f'f2 = {f2}')
"""

if __name__ == '__main__':
    main()
