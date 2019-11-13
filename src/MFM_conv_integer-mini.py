#!/usr/bin/env python
import sys
import pandas as pd
print(sys.version)
print(pd.__version__)

# Hard-coded path name: Change for different machines.
csl_df = pd.read_sas('/home/mbopf/MFMDatasets/CSL_StudyItems/CSLDatasets/csllinkedbypreg.sas7bdat', format = 'sas7bdat', encoding='iso-8859-1')

# Convert any non-float fields to IntegerArray (Int)
# Note than IntegerArrays are an experimental addition in Pandas 0.24. They
# allow integer columns to contain NaN fields like float columns.
#
# This is a rather brute-force technique that loops through every column
# and every row. There's got to be a more efficient way to do it since it 
# takes a long time and uses up a lot of memory.
def convert_to_integer (df):
    for col in df.columns:
        intcol_flag = True
        if df[col].dtype == 'float64':   # Assuming dtype is "float64"
            # If not NaN and the int() value is different from
            # the float value, then we have an actual float.
            # s = df[col].apply(lambda x: pd.notnull(x) and abs(x - int(x)) > 1e-6)
            s = df[col].notnull() & df[col].sub(df[col].round()).abs().gt(1e-6)
            if s.any():
                intcol_flag = False
            # If not a float, change it to an Int based on size
            if intcol_flag:
                if df[col].abs().max() < 127:
                    df[col] = df[col].astype('Int8')
                elif df[col].abs().max() < 32767:
                    df[col] = df[col].astype('Int16')
                else:   # assuming no ints greater than 2147483647 
                    df[col] = df[col].astype('Int32') 
#        print(f"{col} is {df[col].dtype}")
    return df

df = convert_to_integer(csl_df)
csl_df.to_csv('./csl_Int.csv')

#csl_df.to_csv('/home/mbopf/MFMDatasets/CSL_StudyItems/mbopf/csl_all.csv')
#csl_df.to_csv('./csl_all.csv')

