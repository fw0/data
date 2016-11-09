import data.readmissions.readmissions.constants as constants
import pandas as pd
import python_utils.python_utils.caching as caching
import itertools

#@caching.default_cache_fxn_decorator
#@caching.default_read_fxn_decorator
#@caching.default_write_fxn_decorator
def bwh_data():
    return pd.DataFrame.from_csv(constants.bwh_data_file, index_col = None)

def bwh_orig_data():
    return pd.DataFrame.from_csv(constants.bwh_orig_data_file, index_col = None)

#@caching.default_cache_fxn_decorator
#@caching.default_read_fxn_decorator
#@caching.default_write_fxn_decorator
def mgh_data():
    return pd.DataFrame.from_csv(constants.mgh_data_file, index_col = None)

"""
def raw_data_to_monotonic_input(d):
    y_name = 'Readmit'
    ys = list(d[y_name])
    xs = [tuple(row) for (row_name, row) in d[[col for col in d.columns if col != y_name]].iterrows()]
    import monotonic.monotonic.utils as monotonic_utils
    return monotonic_utils.data(range(len(ys)), xs, ys, d.columns)
"""
    
def x_feature_names():
    """
    is the same for both hospitals
    """
    d = mgh_data()
    return d.columns

def bwh_data_history_of_multiple_admissions():
    d = bwh_data()
    return d[d.HistoryofMultipleAdmissions==1][[col for col in d.columns if col != 'HistoryofMultipleAdmissions']]

def bwh_data_NO_history_of_multiple_admissions():
    d = bwh_data()
    return d[d.HistoryofMultipleAdmissions==0][[col for col in d.columns if col != 'HistoryofMultipleAdmissions']]

def bwh_data_substance_abuse():
    d = bwh_data()
    return d[d.SubstanceAbuse==1][[col for col in d.columns if col != 'SubstanceAbuse']]

