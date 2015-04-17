"""
data structure for prostate data will be the same as that accepted by recovery curve code
"""
import recovery_curve.recovery_curve.fxns as recovery_fxns
import numpy as np
import python_utils.python_utils.caching as caching
import pandas as pd
from sklearn.base import TransformerMixin as TransformerMixin
import data.prostate.prostate.constants as constants
import itertools

class data_base(object):

    @classmethod
    def age_init_bin_to_idx(cls, x_ns, age_idx, init_idx):
        for (i,x) in enumerate(x_ns):
            if cls.age_bin(x) == age_idx and cls.init_bin(x) == init_idx:
                return i
        assert False
    
    @classmethod
    def distinct_data(cls):
        Xs, ys_ns = cls.data()
        d = {}
        for ((s,x,t),ys) in itertools.izip(Xs.iterrows(),ys_ns):
            d[(cls.age_bin(x),cls.init_bin(x))] = ((s,x,t),ys)
        print d.keys()
        distinct_Xs_zipped, distinct_ys_ns = zip(*d.values())
        distinct_s_ns, distinct_x_ns, distinct_ts_ns = zip(*distinct_Xs_zipped)
        return recovery_fxns.recovery_X(distinct_s_ns, distinct_x_ns, distinct_ts_ns), distinct_ys_ns
    
class good_sexual_function_data(data_base):
    
    @classmethod
    def age_bin(cls, x):
        if x[4] > 0:
            return 0
        elif x[5] > 0:
            return 1
        else:
            return 2

    @classmethod
    def init_bin(cls, x):
        if x[2] > 0:
            return 0
        elif x[1] > 0:
            return 1
        elif x[3] > 1:
            return 2
        else:
            return 3

    @classmethod
    def num_age_bins(cls):
        return 3

    @classmethod
    def num_init_bins(cls):
        return 4

#    @caching.default_read_fxn_decorator
#    @caching.default_write_fxn_decorator
    @classmethod
    def data(cls):
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
    print data[0].xa
    import pdb
    pdb.set_trace()
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
    import pdb
    return recovery_fxns.recovery_X(\
                                    ys.loc[:,0].values,\
                                    xs.values,\
                                    np.array([np.array(y[y.index != 0].dropna().index) for (idx,y) in ys.iterrows()]),\
                                    ),\
                                    np.array([np.array(y[y.index != 0].dropna()) for (idx,y) in ys.iterrows()])
                                    
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
    post_ys = ys[ys.index > 0].dropna()
    up_to = 0
    for i in xrange(len(post_ys)):
        if post_ys.iloc[i] == 0:
            if up_to+1 >= k:
                return False
            up_to += 1
        else:
            up_to = 0
    if up_to >= k:
        return False
    return True

"""
filtered data used for paper
"""

class data_for_paper(data_base):
    """
    feature order: age_bins(2+1), init_bins(3+1), indicator
    """
    
    @classmethod
    def age_bin(cls, x):
        if x[0] > 0:
            return 0
        elif x[1] > 0:
            return 1
        else:
            return 2

    @classmethod
    def init_bin(cls, x):
        if x[2] > 0:
            return 0
        elif x[3] > 0:
            return 1
        elif x[4] > 0:
            return 2
        else:
            return 3
    
    @classmethod
    def data(cls):
        """
        directly copied over from ipython notebook
        """
        # import modules
        from IPython.display import display_pretty, display_html
        import python_utils.python_utils.basic as basic
        import recovery_curve.recovery_curve.fxns as recovery_fxns
        import pandas as pd
    #    import data.prostate.prostate.fxns as prostate_data
        import itertools
        import matplotlib.pyplot as plt
        import pdb
        import functools
        import python_utils.python_utils.features as features
        import python_utils.python_utils.sklearn_utils as sklearn_utils

        import sys
        prostate_data = sys.modules[__name__]

        # filter xs_data with pipeline
        from sklearn.pipeline import Pipeline
        min_s = 0.1
        xs_filter_pipeline = Pipeline([\
                              ('s_na', sklearn_utils.transform_from_fxn(lambda df: df.dropna(axis=0,subset=['s']))),\
                              ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
                              ])
        xs = prostate_data.xs_with_s_data()
        filtered_xs = xs_filter_pipeline.transform(xs)
        print filtered_xs.shape
        filtered_xs.head()

        # convert to categorical features
        from sklearn.preprocessing import StandardScaler as StandardScaler
        from sklearn import preprocessing
        xs_features = [\
    #                   features.df_feature_from_feature(lambda x:pd.Series({'ind':1}),'s'),\
                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,55,65,None]),'age'),\
                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,.41,.60,.80,None]),'s'),\
                       ]
        xs_catfeat_pipeline = Pipeline([\
                                        ('categorical', sklearn_utils.transform_from_fxn(functools.partial(features.df_from_df_and_df_features, xs_features))),\
        #                                ('normalize', sklearn_utils.transform_from_fxn(lambda df: (df-df.mean(axis=0))/df.std())),\
                                        ])
        xs_catfeat = xs_catfeat_pipeline.transform(filtered_xs)
        print xs_catfeat.shape
        xs_catfeat.head()

        # get ys data and filter it
        ys = prostate_data.sex_ys()
        ys_pipeline = Pipeline([\
                                ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
                                ('filter_low_s', sklearn_utils.pd_filter_transform(lambda y:y.loc[0] > 0.1)),\
                                ('enough_data', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.enough_data, 6))),\
                                ('does_not_improve', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.does_not_improve, 48))),\
                                ('not_k_zeros_in_a_row', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.not_k_zeros_in_a_row, 3))),\
                                ])
        filtered_ys = ys_pipeline.transform(ys)
        print filtered_ys.shape
        filtered_ys.head()

        # merge xs and ys
        joined_xs, joined_ys = sklearn_utils.pd_join_transform().transform(xs_catfeat, filtered_ys)
        joined_xs.shape, joined_ys.shape

        print joined_xs.shape

        # normalize xs
        final_xs, final_ys = sklearn_utils.zip_transform([sklearn_utils.transform_from_fxn(lambda df: ((df-df.mean(axis=0))/df.std())),\
                                                          sklearn_utils.transform_from_fxn(lambda x:x)]).transform(joined_xs, joined_ys)


        # add indicator feature
        final_xs.loc[:,'ind'] = pd.Series(np.ones(final_xs.shape[0]),index=final_xs.index)
        print final_xs.shape, final_ys.shape
        print final_xs.head()



        return prostate_data.xs_and_ys_to_new_format(final_xs, final_ys)

class pids_copied_from_ec2_data_for_paper(data_for_paper):

    @classmethod
    def the_pids(cls):
        return pd.read_csv(constants.from_ec2_pids,header=None,dtype=int).values
        
    @classmethod
    def data(cls):
        """
        the pids to include are copied over from grave in EC2
        """
        # import modules
        from IPython.display import display_pretty, display_html
        import python_utils.python_utils.basic as basic
        import recovery_curve.recovery_curve.fxns as recovery_fxns
        import pandas as pd
    #    import data.prostate.prostate.fxns as prostate_data
        import itertools
        import matplotlib.pyplot as plt
        import pdb
        import functools
        import python_utils.python_utils.features as features
        import python_utils.python_utils.sklearn_utils as sklearn_utils

        import sys
        prostate_data = sys.modules[__name__]

        # filter xs_data with pipeline
        from sklearn.pipeline import Pipeline
        min_s = 0.1
        xs_filter_pipeline = Pipeline([\
#                              ('s_na', sklearn_utils.transform_from_fxn(lambda df: df.dropna(axis=0,subset=['s']))),\
                              ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
                              ])
        xs = prostate_data.xs_with_s_data()
        filtered_xs = xs_filter_pipeline.transform(xs)
        print filtered_xs.shape
        filtered_xs.head()

        # convert to categorical features
        from sklearn.preprocessing import StandardScaler as StandardScaler
        from sklearn import preprocessing
        xs_features = [\
    #                   features.df_feature_from_feature(lambda x:pd.Series({'ind':1}),'s'),\
                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,55,65,None]),'age'),\
                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,.41,.60,.80,None]),'s'),\
                       ]
        xs_catfeat_pipeline = Pipeline([\
                                        ('categorical', sklearn_utils.transform_from_fxn(functools.partial(features.df_from_df_and_df_features, xs_features))),\
        #                                ('normalize', sklearn_utils.transform_from_fxn(lambda df: (df-df.mean(axis=0))/df.std())),\
                                        ])
        xs_catfeat = xs_catfeat_pipeline.transform(filtered_xs)
        print xs_catfeat.shape
        xs_catfeat.head()

        the_pids = cls.the_pids()
        
        # get ys data and filter it
        ys = prostate_data.sex_ys()
        ys_pipeline = Pipeline([\
                                ('pid_filter',sklearn_utils.pd_filter_index_transform(lambda pid: pid in the_pids))
#                                ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
#                                ('filter_low_s', sklearn_utils.pd_filter_transform(lambda y:y.loc[0] > 0.1)),\
#                                ('enough_data', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.enough_data, 6))),\
#                                ('does_not_improve', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.does_not_improve, 48))),\
#                                ('not_k_zeros_in_a_row', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.not_k_zeros_in_a_row, 3))),\
                                ])
        filtered_ys = ys_pipeline.transform(ys)
        print filtered_ys.shape
        filtered_ys.head()

        # merge xs and ys
        joined_xs, joined_ys = sklearn_utils.pd_join_transform().transform(xs_catfeat, filtered_ys)
        joined_xs.shape, joined_ys.shape

        print joined_xs.shape

        # normalize xs
        final_xs, final_ys = sklearn_utils.zip_transform([sklearn_utils.transform_from_fxn(lambda df: ((df-df.mean(axis=0))/df.std())),\
                                                          sklearn_utils.transform_from_fxn(lambda x:x)]).transform(joined_xs, joined_ys)


        # add indicator feature
        final_xs.loc[:,'ind'] = pd.Series(np.ones(final_xs.shape[0]),index=final_xs.index)
        print final_xs.shape, final_ys.shape
        print final_xs.head()



        return prostate_data.xs_and_ys_to_new_format(final_xs, final_ys)

class not_pids_copied_from_ec2_data_for_paper(data_for_paper):

    @classmethod
    def the_pids(cls):
        return pd.read_csv(constants.from_ec2_pids,header=None,dtype=int).values
        
    @classmethod
    def data(cls):
        """
        the pids to include are copied over from grave in EC2
        """
        # import modules
        from IPython.display import display_pretty, display_html
        import python_utils.python_utils.basic as basic
        import recovery_curve.recovery_curve.fxns as recovery_fxns
        import pandas as pd
    #    import data.prostate.prostate.fxns as prostate_data
        import itertools
        import matplotlib.pyplot as plt
        import pdb
        import functools
        import python_utils.python_utils.features as features
        import python_utils.python_utils.sklearn_utils as sklearn_utils

        import sys
        prostate_data = sys.modules[__name__]

        # filter xs_data with pipeline
        from sklearn.pipeline import Pipeline
        min_s = 0.1
        xs_filter_pipeline = Pipeline([\
#                              ('s_na', sklearn_utils.transform_from_fxn(lambda df: df.dropna(axis=0,subset=['s']))),\
                              ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
                              ])
        xs = prostate_data.xs_with_s_data()
        filtered_xs = xs_filter_pipeline.transform(xs)
        print filtered_xs.shape
        filtered_xs.head()

        # convert to categorical features
        from sklearn.preprocessing import StandardScaler as StandardScaler
        from sklearn import preprocessing
        xs_features = [\
    #                   features.df_feature_from_feature(lambda x:pd.Series({'ind':1}),'s'),\
                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,55,65,None]),'age'),\
                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,.41,.60,.80,None]),'s'),\
                       ]
        xs_catfeat_pipeline = Pipeline([\
                                        ('categorical', sklearn_utils.transform_from_fxn(functools.partial(features.df_from_df_and_df_features, xs_features))),\
        #                                ('normalize', sklearn_utils.transform_from_fxn(lambda df: (df-df.mean(axis=0))/df.std())),\
                                        ])
        xs_catfeat = xs_catfeat_pipeline.transform(filtered_xs)
        print xs_catfeat.shape
        xs_catfeat.head()

        the_pids = cls.the_pids()
        
        # get ys data and filter it
        ys = prostate_data.sex_ys()
        ys_pipeline = Pipeline([\
                                ('pid_filter',sklearn_utils.pd_filter_index_transform(lambda pid: pid not in the_pids))
#                                ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
#                                ('filter_low_s', sklearn_utils.pd_filter_transform(lambda y:y.loc[0] > 0.1)),\
#                                ('enough_data', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.enough_data, 6))),\
#                                ('does_not_improve', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.does_not_improve, 48))),\
#                                ('not_k_zeros_in_a_row', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.not_k_zeros_in_a_row, 3))),\
                                ])
        filtered_ys = ys_pipeline.transform(ys)
        print filtered_ys.shape
        filtered_ys.head()

        # merge xs and ys
        joined_xs, joined_ys = sklearn_utils.pd_join_transform().transform(xs_catfeat, filtered_ys)
        joined_xs.shape, joined_ys.shape

        print joined_xs.shape

        # normalize xs
        final_xs, final_ys = sklearn_utils.zip_transform([sklearn_utils.transform_from_fxn(lambda df: ((df-df.mean(axis=0))/df.std())),\
                                                          sklearn_utils.transform_from_fxn(lambda x:x)]).transform(joined_xs, joined_ys)


        # add indicator feature
        final_xs.loc[:,'ind'] = pd.Series(np.ones(final_xs.shape[0]),index=final_xs.index)
        print final_xs.shape, final_ys.shape
        print final_xs.head()



        return prostate_data.xs_and_ys_to_new_format(final_xs, final_ys)


class pids_copied_from_ec2_data_for_paper_age_only(data_for_paper):

    @classmethod
    def age_bin_to_idx(cls, x_ns, age_idx):
        for (i,x) in enumerate(x_ns):
            if cls.age_bin(x) == age_idx:
                return i
        assert False
    
    @classmethod
    def age_bin(cls, x):
        if x[0] > 0:
            return 0
        elif x[1] > 0:
            return 1
        else:
            return 2

    @classmethod
    def num_age_bins(cls):
        return 3
    
    @classmethod
    def the_pids(cls):
        return pd.read_csv(constants.from_ec2_pids,header=None,dtype=int).values
        
    @classmethod
    def data(cls):
        """
        the pids to include are copied over from grave in EC2
        """
        # import modules
        from IPython.display import display_pretty, display_html
        import python_utils.python_utils.basic as basic
        import recovery_curve.recovery_curve.fxns as recovery_fxns
        import pandas as pd
    #    import data.prostate.prostate.fxns as prostate_data
        import itertools
        import matplotlib.pyplot as plt
        import pdb
        import functools
        import python_utils.python_utils.features as features
        import python_utils.python_utils.sklearn_utils as sklearn_utils

        import sys
        prostate_data = sys.modules[__name__]

        # filter xs_data with pipeline
        from sklearn.pipeline import Pipeline
        min_s = 0.1
        xs_filter_pipeline = Pipeline([\
#                              ('s_na', sklearn_utils.transform_from_fxn(lambda df: df.dropna(axis=0,subset=['s']))),\
                              ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
                              ])
        xs = prostate_data.xs_with_s_data()
        filtered_xs = xs_filter_pipeline.transform(xs)
        print filtered_xs.shape
        filtered_xs.head()

        # convert to categorical features
        from sklearn.preprocessing import StandardScaler as StandardScaler
        from sklearn import preprocessing
        xs_features = [\
    #                   features.df_feature_from_feature(lambda x:pd.Series({'ind':1}),'s'),\
                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,55,65,None]),'age'),\
#                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,.41,.60,.80,None]),'s'),\
                       ]
        xs_catfeat_pipeline = Pipeline([\
                                        ('categorical', sklearn_utils.transform_from_fxn(functools.partial(features.df_from_df_and_df_features, xs_features))),\
        #                                ('normalize', sklearn_utils.transform_from_fxn(lambda df: (df-df.mean(axis=0))/df.std())),\
                                        ])
        xs_catfeat = xs_catfeat_pipeline.transform(filtered_xs)
        print xs_catfeat.shape
        xs_catfeat.head()

        the_pids = cls.the_pids()
        
        # get ys data and filter it
        ys = prostate_data.sex_ys()
        ys_pipeline = Pipeline([\
                                ('pid_filter',sklearn_utils.pd_filter_index_transform(lambda pid: pid in the_pids))
#                                ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
#                                ('filter_low_s', sklearn_utils.pd_filter_transform(lambda y:y.loc[0] > 0.1)),\
#                                ('enough_data', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.enough_data, 6))),\
#                                ('does_not_improve', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.does_not_improve, 48))),\
#                                ('not_k_zeros_in_a_row', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.not_k_zeros_in_a_row, 3))),\
                                ])
        filtered_ys = ys_pipeline.transform(ys)
        print filtered_ys.shape
        filtered_ys.head()

        # merge xs and ys
        joined_xs, joined_ys = sklearn_utils.pd_join_transform().transform(xs_catfeat, filtered_ys)
        joined_xs.shape, joined_ys.shape

        print joined_xs.shape

        # normalize xs
        final_xs, final_ys = sklearn_utils.zip_transform([sklearn_utils.transform_from_fxn(lambda df: ((df-df.mean(axis=0))/df.std())),\
                                                          sklearn_utils.transform_from_fxn(lambda x:x)]).transform(joined_xs, joined_ys)


        # add indicator feature
        final_xs.loc[:,'ind'] = pd.Series(np.ones(final_xs.shape[0]),index=final_xs.index)
        print final_xs.shape, final_ys.shape
        print final_xs.head()



        return prostate_data.xs_and_ys_to_new_format(final_xs, final_ys)


class pids_copied_from_ec2_data_for_paper_init_only(data_for_paper):

    @classmethod
    def init_bin_to_idx(cls, x_ns, init_idx):
        for (i,x) in enumerate(x_ns):
            if cls.init_bin(x) == init_idx:
                return i
        assert False
    
    @classmethod
    def init_bin(cls, x):
        if x[0] > 0:
            return 0
        elif x[1] > 0:
            return 1
        elif x[2] > 0:
            return 2
        elif x[3] > 0:
            return 3
        assert False

    @classmethod
    def num_init_bins(cls):
        return 4
    
    @classmethod
    def the_pids(cls):
        return pd.read_csv(constants.from_ec2_pids,header=None,dtype=int).values
        
    @classmethod
    def data(cls):
        """
        the pids to include are copied over from grave in EC2
        """
        # import modules
        from IPython.display import display_pretty, display_html
        import python_utils.python_utils.basic as basic
        import recovery_curve.recovery_curve.fxns as recovery_fxns
        import pandas as pd
    #    import data.prostate.prostate.fxns as prostate_data
        import itertools
        import matplotlib.pyplot as plt
        import pdb
        import functools
        import python_utils.python_utils.features as features
        import python_utils.python_utils.sklearn_utils as sklearn_utils

        import sys
        prostate_data = sys.modules[__name__]

        # filter xs_data with pipeline
        from sklearn.pipeline import Pipeline
        min_s = 0.1
        xs_filter_pipeline = Pipeline([\
#                              ('s_na', sklearn_utils.transform_from_fxn(lambda df: df.dropna(axis=0,subset=['s']))),\
                              ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
                              ])
        xs = prostate_data.xs_with_s_data()
        filtered_xs = xs_filter_pipeline.transform(xs)
        print filtered_xs.shape
        filtered_xs.head()

        # convert to categorical features
        from sklearn.preprocessing import StandardScaler as StandardScaler
        from sklearn import preprocessing
        xs_features = [\
    #                   features.df_feature_from_feature(lambda x:pd.Series({'ind':1}),'s'),\
#                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,55,65,None]),'age'),\
                       features.df_feature_from_feature(features.bins_feature.from_boundaries([0,.41,.60,.80,None]),'s'),\
                       ]
        xs_catfeat_pipeline = Pipeline([\
                                        ('categorical', sklearn_utils.transform_from_fxn(functools.partial(features.df_from_df_and_df_features, xs_features))),\
        #                                ('normalize', sklearn_utils.transform_from_fxn(lambda df: (df-df.mean(axis=0))/df.std())),\
                                        ])
        xs_catfeat = xs_catfeat_pipeline.transform(filtered_xs)
        print xs_catfeat.shape
        xs_catfeat.head()

        the_pids = cls.the_pids()
        
        # get ys data and filter it
        ys = prostate_data.sex_ys()
        ys_pipeline = Pipeline([\
                                ('pid_filter',sklearn_utils.pd_filter_index_transform(lambda pid: pid in the_pids))
#                                ('prostatectomy_filter', sklearn_utils.transform_from_fxn(prostate_data.filter_df_by_treatment(prostate_data.filter_df_by_treatment.prostatectomy))),\
#                                ('filter_low_s', sklearn_utils.pd_filter_transform(lambda y:y.loc[0] > 0.1)),\
#                                ('enough_data', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.enough_data, 6))),\
#                                ('does_not_improve', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.does_not_improve, 48))),\
#                                ('not_k_zeros_in_a_row', sklearn_utils.pd_filter_transform(functools.partial(prostate_data.not_k_zeros_in_a_row, 3))),\
                                ])
        filtered_ys = ys_pipeline.transform(ys)
        print filtered_ys.shape
        filtered_ys.head()

        # merge xs and ys
        joined_xs, joined_ys = sklearn_utils.pd_join_transform().transform(xs_catfeat, filtered_ys)
        joined_xs.shape, joined_ys.shape

        print joined_xs.shape

        # normalize xs
        final_xs, final_ys = sklearn_utils.zip_transform([sklearn_utils.transform_from_fxn(lambda df: ((df-df.mean(axis=0))/df.std())),\
                                                          sklearn_utils.transform_from_fxn(lambda x:x)]).transform(joined_xs, joined_ys)


        # add indicator feature
        final_xs.loc[:,'ind'] = pd.Series(np.ones(final_xs.shape[0]),index=final_xs.index)
        print final_xs.shape, final_ys.shape
        print final_xs.head()



        return prostate_data.xs_and_ys_to_new_format(final_xs, final_ys)
