# Code credit: Shaked Zychlinski:
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
import pandas as pd
import numpy as np
import math
import scipy.stats as ss
from collections import Counter

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def cramers_v_df(df) -> pd.DataFrame:
    numcols = df.shape[1]
    cdf = pd.DataFrame(np.zeros((numcols, numcols)), index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            cv = cramers_v(df[col1], df[col2])
            cdf.loc[col1, col2] = cv
    return cdf


# Code credit: Shaked Zychlinski:
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def conditional_entropy(x, y):
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def theils_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        # return 1  # MWB - Changing this since perfect correlation doesn't make sense
        return 0
    else:
        return (s_x - s_xy) / s_x


def theils_u_df(df) -> pd.DataFrame:
    numcols = df.shape[1]
    udf = pd.DataFrame(np.zeros((numcols, numcols)), index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            tu = theils_u(df[col1], df[col2])
            udf.loc[col1, col2] = tu
    return udf

