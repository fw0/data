import data.problem_props_data.problem_props_data.constants as constants
import pandas as pd
from python_utils.python_utils.sklearn_utils import myTransformerMixin
import numpy as np
import pdb

def parcels_data():
    raw = pd.read_csv(constants.parcels_file)[['X','Y','TLID']]
    raw.columns = ['lngs', 'lats', 'TLID']
    raw.index = pd.read_csv(constants.parcels_file)['Property_ID']
    return raw

def raw_data():
    path = '%s/%s' % (constants.data_folder, 'processed_911_data.csv')
    call_data = pd.DataFrame.from_csv(path, index_col=0, header=0)
    _parcels_data = parcels_data()
    return call_data.merge(_parcels_data, left_on='SAM_ID', right_index=True)

def zero_count_data():
    X_counts_df, y_counts_series = convert_to_space_count().transform(raw_data())
    _parcels_data = parcels_data()
    merged = X_counts_df.merge(_parcels_data, left_index=True, right_index=True, how='outer')
    merged = merged[['lngs_y','lats_y']]
    merged.columns = ['lngs','lats']
    merged = merged.merge(pd.DataFrame({'counts':y_counts_series}), left_index=True, right_index=True,how='outer')
    merged.counts.fillna(0, inplace=True)
    merged = merged[merged.counts==0]
    return merged[['lngs','lats']]
    
min_lng, max_lng, min_lat, max_lat = -72.959158847796786, -72.917530446490503, 36.683225718858601, 36.722725425165294





class filter_by_num_calls(myTransformerMixin):
    
    def __init__(self, the_min, the_max):
        self.the_min, self.the_max = the_min, the_max

    def transform(self, data):
        print self.the_min, self.the_max
        def f(x):
            #print len(x), self.the_min, self.the_max
#            ans = False
            ans = (len(x) < self.the_max) and (len(x) > self.the_min)
            #print ans
            #if len(x) < self.the_min:
            #    pdb.set_trace()
            return ans
        return data.groupby('SAM_ID').filter(f)


class filter_by_types(myTransformerMixin):

    def __init__(self, types):
        self.types = types

    def transform(self, df):
        return df.loc[df.type.apply(lambda x: x in self.types)]

            
class filter_df_by_percentile(myTransformerMixin):

    def __init__(self, col, min_percentile, max_percentile):
        self.col, self.min_percentile, self.max_percentile = col, min_percentile, max_percentile

    def transform(self, df):
        low = np.percentile(df[self.col], self.min_percentile)
        high = np.percentile(df[self.col], self.max_percentile)
        return df[(df[self.col]>low) & (df[self.col]<high)]

        
class filter_df_by_sampling(myTransformerMixin):

    def __init__(self, col, prop):
        self.col, self.prop = col, prop

    def transform(self, df):
        if self.col == 'index':
            return df.groupby(df.index).filter(lambda x: np.random.uniform() < self.prop)
        else:
            return df.groupby(self.col).filter(lambda x: np.random.uniform() < self.prop)

            
class convert_to_space_count(myTransformerMixin):
    """
    output will have index, because eventually will use index to filter raw_data
    TODO: in process data, read in SAMid instead of propid, and group by SAMid
    """
    def transform(self, data):
        df = data.groupby('SAM_ID').agg({'SAM_ID':len,'lats':lambda x:x.iloc[0], 'lngs':lambda x:x.iloc[0]})
        X = df[['lngs','lats']]
        y = df['SAM_ID']
        return X, y


#def add_0_counts_to_counts(X_counts_df, y_counts, parcels_data):
    

        
class convert_to_type_counts(myTransformerMixin):

    def transform(self, data):
        d = data.groupby('SAM_ID')['type'].apply(lambda x:x.value_counts())
        return pd.DataFrame.from_items([(idx,d[idx]) for idx in d.index.levels[0]]).fillna(0).T
        
