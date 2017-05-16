import constants
import pandas as pd
#import data.NSW.NSW.fxns as data_fxns
import itertools
import python_utils.python_utils.features as features
from sklearn.pipeline import Pipeline
import python_utils.python_utils.sklearn_utils as sklearn_utils
import functools
import numpy as np

def NSW_data():
#<<<<<<< HEAD
#    ans = pd.DataFrame.from_csv(constants.file_path)
#    ans['white'] = map(int,(ans.black + ans.hispan == 0))
#    return ans
#=======
    return pd.DataFrame.from_csv(constants.file_path)

def binarized_NSW_data(trim=None):
    # read in data
    data = NSW_data()

    # filter/transform data
    def scale_salary(df):
        df.re78 = df.re78 / 10000
        return df
    preprocess_pipeline = Pipeline([
        ('no_zeros', sklearn_utils.transform_from_fxn(lambda df: df[df.re78>.001])),
        ('scale_salary',sklearn_utils.transform_from_fxn(scale_salary))
        ])
    preprocessed_data = preprocess_pipeline.transform(data)

    # convert to categorical features
    from sklearn.preprocessing import StandardScaler as StandardScaler
    from sklearn import preprocessing
    df_bin_feats = [
        features.df_feature_from_feature(features.bins_feature.from_boundaries([0,25,40,None],drop=False),'age'),
        features.df_feature_from_feature(features.bins_feature.from_boundaries([0,8,12,None],drop=False),'educ'),
        features.df_feature_from_feature(features.bins_feature.from_boundaries([0,1,7500,None],drop=False),'re74'),
#               features.df_feature_from_feature(features.bins_feature.from_boundaries([.0001,.7500,None],drop=False),'re74'),
        ]
    iden_feats = [
        features.df_feature_from_feature(lambda x:pd.Series([x],index=['']), 'black'),
        features.df_feature_from_feature(lambda x:pd.Series([x],index=['']), 'hispan'),
        features.df_feature_from_feature(lambda x:pd.Series([x],index=['']), 'married'),
        features.df_feature_from_feature(lambda x:pd.Series([x],index=['']), 'nodegree'),
        #features.df_feature_from_feature(lambda x:pd.Series([x],index=['']), 're74'),
        ]
    bin_feats_pipeline = Pipeline([
        ('categorical', sklearn_utils.transform_from_fxn(functools.partial(features.df_from_df_and_df_features, df_bin_feats+iden_feats))),
#                                ('normalize', sklearn_utils.transform_from_fxn(lambda df: (df-df.mean(axis=0))/df.std())),\
        ])
    bin_feats_data = bin_feats_pipeline.transform(preprocessed_data)
    print bin_feats_data.shape
    bin_feats_data.head()

    final_feats_data = pd.concat([
        bin_feats_data,
        pd.DataFrame({'indicator':np.ones(len(bin_feats_data))},index=bin_feats_data.index),
        ], join='inner', axis=1)


    # define data for model fitter
    T_ns = preprocessed_data.treat.values
    x_ns = final_feats_data.values
    y_ns = preprocessed_data.re78.values
    K = x_ns.shape[1]
    x_names = final_feats_data.columns

    if not (trim is None):
        from causalinference import CausalModel
        cm = CausalModel(y_ns, T_ns, x_ns)
        cm.est_propensity()
        cm.cutoff = trim
        cm.trim()
        import pdb
        pdb.set_trace()
    
    return x_ns, x_names, T_ns, y_ns
>>>>>>> origin/master
