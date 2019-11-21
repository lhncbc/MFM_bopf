##############################################################################
# Convert all float columns that are actually integers with some NaN values
##############################################################################
import pandas as pd
def convert_integer(df):
    type_dict = {}
    for col in df.columns:
        intcol_flag = True
        if df[col].dtype == 'float64':
            # Inner loop makes this very slow, but can't find a vectorized solution
            for val in df[col]:
                if pd.notnull(val) and abs(val - int(val)) > 1e-6:
                    intcol_flag = False
                    break;
            if intcol_flag:
                if df[col].abs().max() < 127:
                    df[col] = df[col].astype('Int8')
                elif df[col].abs().max() < 32767:
                    df[col] = df[col].astype('Int16')
                else:   # assuming no ints greater than 2147483647 
                    df[col] = df[col].astype('Int32') 
#        print(f"{col} is {df[col].dtype}")
        type_dict[col] = df[col].dtype
    return type_dict


##############################################################################
# Read a subset of a large file.
##############################################################################
# Note hard-coded default filename (temporary)
# "types-file" is assumed to be a pickled Python dictionary of column names to Python datatypes.
def subset_csv(filename='../data/csl/preg_link.csv', rows=100, columns=10, random=False, types_file=False):
    if random:
        print("Random not yet implemented; first rows and columns used")
    if types_file:
        import pickle
        with open(types_file, 'rb') as file:
            csl_types = pickle.load(file)
        df = pd.read_csv(filename, index_col=0,  nrows=rows, usecols=range(0, columns+1),
                         header=0, skip_blank_lines=True, dtype=csl_types)
    else:
        df = pd.read_csv(filename, index_col=0,  nrows=rows, usecols=range(0, columns+1),
                         header=0, skip_blank_lines=True)
    return df
