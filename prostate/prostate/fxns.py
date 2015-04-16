"""
data structure for prostate data will be the same as that accepted by recovery curve code
"""
import recovery_curve.recovery_curve.fxns as recovery_fxns
import numpy as np
import python_utils.python_utils.caching as caching
import pandas as pd
from sklearn.base import TransformerMixin as TransformerMixin
import data.prostate.prostate.constants as constants

@caching.default_read_fxn_decorator
@caching.default_write_fxn_decorator
def good_sexual_function_data():
    """
    the data used for paper
    """
    import pickle
    return pickle.load(open(constants.old_good_sexual_function_data_pickle_file,'rb'))
    from recovery_curve_old.hard_coded_objects.go_two_real_data import data as d
    return data_to_new_format(d)

def all_sexual_function_data():
    """
    the data used for paper.
    actually this is the same as 'good' data, so this is an error
    """
    from recovery_curve_old.hard_coded_objects.go_two_real_data_all import data as d
    return data_to_new_format(d)

def data_to_new_format(data):
    import recovery_curve.recovery_curve.fxns as recovery_fxns
    return recovery_fxns.recovery_X(\
                                    np.array([datum.s for datum in data]),\
                                    np.array([datum.xa for datum in data]),\
                                    np.array([np.array(datum.ys.index) for datum in data]),\
                                    ),\
                                    np.array([np.array(datum.ys) for datum in data])


def xs_and_ys_to_new_format(xs, ys):
    """
    xs, ys are dataframes
    will have to drop the observations at time 0
    """
    return recovery_fxns.recovery_X(\
                                    ys[0],\
                                    xs.values,\
                                    np.array([np.array(y.dropna().index) for (idx,y) in ys.iterrows()]),\
                                    ),\
                                    np.array([np.array(y.dropna()) for (idx,y) in ys.iterrows()])
                                    
def xs_data():
    import data.prostate.prostate.constants as constants
    xs = pd.DataFrame.from_csv(constants.xs_file,index_col=0).T
    xs.index = xs.index.astype(int)
    return xs
    
def xs_with_s_data():
    xs = xs_data()
    ys = sex_ys()
    xs['s'] = ys[0]
    return xs

def sex_ys():
    import data.prostate.prostate.constants as constants
    d = pd.DataFrame.from_csv(constants.sexual_function_data_file,index_col=0).T
    d.index = d.index.astype(int)
    return d


class filter_df_by_treatment(object):

    prostatectomy = 0
    
    def __init__(self, treatment):
        self.treatment = treatment

    def __call__(self, df):
        if self.treatment == filter_df_by_treatment.prostatectomy:
            return df.loc[(df.index >= 30000) & (df.index < 40000)]
        assert False
        
def prostatectomy_sex_ys(min_s=0):
    d = sex_ys()
    import pdb
    return d.loc[(d.index >= 30000) & (d.index < 40000) & (d[0] > min_s)]

"""
some boolean functions of original ys dataframe
"""
def enough_data(cutoff, ys):
    post_ys = ys[ys.index > 0]
    #print post_ys, len(post_ys.dropna()), cutoff
    return len(post_ys.dropna()) >= cutoff

def does_not_improve(time, ys):
    post_ys = ys[ys.index > 0]
    decay_fit_f = recovery_fxns.fit_decay_f(post_ys.index, post_ys)
    s = ys[0]
    return decay_fit_f(time) < s

def not_k_zeros_in_a_row(k, ys):
    ans1 = True
    post_ys = ys[ys.index > 0].dropna()
    up_to = 0
    for i in xrange(len(post_ys)):
        if post_ys.iloc[i] == 0:
            if up_to+1 >= k:
                ans1 = False
            up_to += 1
        else:
            up_to = 0
    if up_to >= k:
        ans1 = False

    """
    this was the old code, but it worked with dataframes with NA
    ans2 = True
    for i in range(len(ys)-k):
        if sum([ys.iloc[x] < 0.01001 for x in range(i,i+k)]) == k:
            ans2 = False
    """
    return ans1
    try:
        assert ans1 == ans2
    except:
        print ans1, ans2
        print ys
        pdb.set_trace()
    return ans1
    return True
