import pandas as pd
import pickle

## Custom libs
from conf import PathConfig

with open(PathConfig.DATA_PATH / 'sid2name.pkl', 'rb') as p:
    SID2NAME = pickle.load(p)

def get_fdr_last(df, col='Close'):
    assert col in ['Close', 'Open', 'High', 'Low', 'Change']
    
    row = df.iloc[-1]
    
    return row[col]

def sid2name(sid, none_if_not_found=True):
    assert isinstance(sid, str)
    
    try:
        return SID2NAME[sid]
    except KeyError as e:
        if none_if_not_found:
            return None
        else:
            raise e