import data.mushroom.mushroom.constants as constants
import pandas as pd
import python_utils.python_utils.caching as caching
import itertools

y_name = 'poisonous'

#@caching.default_cache_fxn_decorator
#@caching.default_read_fxn_decorator
#@caching.default_write_fxn_decorator
def mushroom_data():
    d = pd.DataFrame.from_csv(constants.mushroom_data_file, index_col = None)
    d[y_name] = pd.Series(d[y_name] == 1, dtype=int)
    return d

"""
def raw_data_to_monotonic_input(d):
    ys = d[y_name] == 1
    xs = [tuple(row) for (row_name, row) in d[[col for col in d.columns if col != y_name]].iterrows()]
    import monotonic.monotonic.utils as monotonic_utils
    return monotonic_utils.data(range(len(ys)), xs, ys)
"""
def spect_data():
    return pd.DataFrame.from_csv(constants.spect_data_file, index_col = None)

def vote_data():
    return pd.DataFrame.from_csv(constants.vote_data_file, index_col = None)

def mammo_data():
    return pd.DataFrame.from_csv(constants.mammo_data_file, index_col = None)

def spam_data():
    d = pd.DataFrame.from_csv(constants.spam_data_file, index_col = None)
    d.iloc[:,-1] = pd.Series(d.iloc[:,-1] == 1, dtype=int)
    return d

def haberman_data():
    return pd.DataFrame.from_csv(constants.haberman_data_file, index_col = None)

def breast_data():
    return pd.DataFrame.from_csv(constants.breast_data_file, index_col = None)

def cars_data():
    return pd.DataFrame.from_csv(constants.cars_data_file, index_col = None)

def tictactoe_data():
    return pd.DataFrame.from_csv(constants.tictactoe_data_file, index_col = None)

def whitewine_data():
    d = pd.DataFrame.from_csv(constants.whitewine_data_file, index_col = None, sep=';')
    return d.iloc[:,:-1].values, d.iloc[:,-1].values
