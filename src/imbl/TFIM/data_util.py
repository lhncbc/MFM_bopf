
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
