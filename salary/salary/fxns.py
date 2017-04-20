import pandas as pd
import constants
import pdb
import itertools
import python_utils.python_utils.features as features
from sklearn.pipeline import Pipeline
import python_utils.python_utils.sklearn_utils as sklearn_utils
import functools
import numpy as np
import python_utils.python_utils.caching as caching

def salary_df(s=slice(0,None)):
    """
    hardcode list of (name,(start_pos,width)) tuples
    """
    hardcoded = [\
                 ('gender',(20,1)),\
                 ('age',(15,2)),\
                 ('workclass',(202,1)),\
                 ('education',(22,2)),\
                 ('marital',(17,1)),\
                 ('industry',(155,2)),\
                 ('occupation',(159,2)),\
                 ('race',(25,1)),\
                 ('union',(139,1)),\
                 ('fulltime',(183,1)),\
                 ('tax',(657,1)),\
                 ('household_summary',(34,1)),\
                 ('employer_size',(226,1)),\
                 ('country',(722,3)),\
                 ('citizenship',(733,1)),\
                 ('weeks_worked',(171,2)),\
                 ('hourly_wage',(131,4)),\
                 ]
    
    f = open(constants.file_path,'r')
    d = {}
    for (i,line) in enumerate(f):
        if line[0] == '3':
            d[i] = {name:int(line[(start_pos-1):(start_pos-1)+width]) for (name,(start_pos,width)) in hardcoded}
    print 'i', i
    return pd.DataFrame(d).T.iloc[s,:]

@caching.default_read_fxn_decorator()
@caching.default_write_fxn_decorator()
def binarized_salary_df(s=slice(0,None)):
    
    raw_data = salary_df(s)

    # filter raw data
    def scale_salary(df):
        df.hourly_wage = df.hourly_wage / 100.
        return df
    rows_to_keep = 2000
    preprocess_pipeline = Pipeline([
        ('has_salary', sklearn_utils.transform_from_fxn(lambda df: df[df.hourly_wage>.001])),
        ('scale_salary',sklearn_utils.transform_from_fxn(scale_salary)),
#        ('truncate',sklearn_utils.transform_from_fxn(lambda df:df.iloc[0:rows_to_keep,:])),
        ])
    data = preprocess_pipeline.transform(raw_data)

    # extract data for categorical features.  this is used as input for binary features from categorical ones, as well as gender
    cat_feats_columns = ['citizenship','country','education','employer_size','fulltime','gender','household_summary','industry','marital','occupation','race','tax','union','workclass']
    cat_feats_data = data[cat_feats_columns]

    # for categorical features, remap the levels to get another dataframe, then drop_NA
    dff = features.df_feature_from_feature
    remap_feats = [
        dff(features.map_feature({
            features.equals_bin(1):'M',
            features.equals_bin(2):'F'
            }),'gender'),
        dff(features.map_feature({
            features.contains_bin([1,2,3]):'citizen',
            features.equals_bin(4):'naturalized',
            features.equals_bin(5):'not_citizen',
            }),'citizenship'),
        dff(features.map_feature({
            features.interval_bin(31,35):'less_hs',
            features.interval_bin(35,39):'some_hs',
            features.equals_bin(39):'hs_grad',
            features.equals_bin(40):'some_college',
            features.interval_bin(41,43):'associates',
            features.equals_bin(43):'bachelors',
            features.equals_bin(44):'masters',
            features.equals_bin(45):'prof_school',
            features.equals_bin(46):'phd',
            }),'education'),
        dff(features.map_feature({
            features.equals_bin(1):'None_10',
            features.equals_bin(2):'10_24',
            features.equals_bin(3):'25_99',
            features.equals_bin(4):'100_499',
            features.equals_bin(5):'500_999',
            features.equals_bin(6):'1000_None',
            }),'employer_size'),
        dff(features.map_feature({
            features.equals_bin(1):'part_time',
            features.equals_bin(2):'full_time',
            }),'fulltime'),
        dff(features.map_feature({
            features.equals_bin(1):'householder',
            features.equals_bin(2):'spouse_of_householder',
            features.equals_bin(3):'child_under_18_single',
            features.equals_bin(4):'child_under_18_married',
            features.equals_bin(5):'child_over_18',
            features.equals_bin(6):'other_relationship',
            features.equals_bin(7):'group_quarters',
            }),'household_summary'),
        dff(features.map_feature({
            features.equals_bin(1):'agriculture',
            features.equals_bin(2):'mining',
            features.equals_bin(3):'construction',
            features.equals_bin(4):'manufacturing_durable',
            features.equals_bin(5):'manufacturing_nondurable',
            features.equals_bin(6):'transportation',
            features.equals_bin(7):'communications',
            features.equals_bin(8):'utilities',
            features.equals_bin(9):'wholesale_trade',
            features.equals_bin(10):'retail_trade',
            features.equals_bin(11):'finance_insurance',
            features.equals_bin(12):'private_household_service',
            features.equals_bin(13):'business_repair',
            features.equals_bin(14):'other_private_service',
            features.equals_bin(15):'entertainment',
            features.equals_bin(16):'hospital',
            features.equals_bin(17):'medical',
            features.equals_bin(18):'educational',
            features.equals_bin(19):'social_service',
            features.equals_bin(20):'other_service',
            features.equals_bin(21):'forestry_fisheries',
            features.equals_bin(22):'public_admin',
        }),'industry'),
#              dff(features.map_feature({\
#                                        features.interval_bin(10,31):'agriculture',\
#                                        features.interval_bin(40,51):'mining',\
#                                        features.equals_bin(60):'construction',\
#                                        features.interval_bin(230,393):'manufacturing_durable',\
#                                        features.interval_bin(100,223):'manufacturing_non_durable',\
#                                        features.interval_bin(400,443):'transportation',\
#                                        features.interval_bin(440,443):'communication_utilities',\
#                                        features.interval_bin(450,473):'other_utilities',\
#                                        features.interval_bin(500,572):'wholesale_trade',\
#                                        features.interval_bin(580,692):'retail_trade',\
#                                        features.interval_bin(700,713):'finance_real_estate',\
#                                        features.equals_bin(761):'private_service',\
#                                        features.interval_bin(721,761):'business_service',\
#                                        features.interval_bin(762,792):'personal_service',\
#                                        features.interval_bin(800,811):'entertainment_service',\
#                                        features.equals_bin(831):'hospital_service',\
#                                        features.union_bin([features.interval_bin(812,831),features.interval_bin(832,841)]):'medical_service',\
#                                        features.interval_bin(842,861):'educational_service',\
#                                        features.interval_bin(861,872):'social_service',\
#                                        features.union_bin([features.equals_bin(841),features.interval_bin(872,894)]):'other_service',\
#                                        features.interval_bin(900,933):'public_administration',\
#                                       }),'industry'),\
        dff(features.map_feature({
            features.equals_bin(1):'civilian_spouse',
            features.equals_bin(2):'armed_spouse',
            features.equals_bin(3):'married_separated',
            features.equals_bin(4):'widowed',
            features.equals_bin(5):'divorced',
            features.equals_bin(6):'separated',
            features.equals_bin(7):'never_married',
        }),'marital'),
        dff(features.map_feature({
            features.equals_bin(1):'executive',
            features.equals_bin(2):'professional_specialty',
            features.equals_bin(3):'technician',
            features.equals_bin(4):'sales',
            features.equals_bin(5):'administrative',
            features.equals_bin(6):'private_service',
            features.equals_bin(7):'protective_service',
            features.equals_bin(8):'other_service',
            features.equals_bin(9):'skilled_tech_service',
            features.equals_bin(10):'factory',
            features.equals_bin(11):'transportation',
            features.equals_bin(12):'handler_equip_cleaner',
            features.equals_bin(13):'farming_forestry',
            features.equals_bin(14):'armed_forces',
        }),'occupation'),
        dff(features.map_feature({
            features.equals_bin(1):'white',
            features.equals_bin(2):'black',
            features.equals_bin(3):'american_indian',
            features.equals_bin(4):'asian',
            features.equals_bin(5):'other',
        }),'race'),
        dff(features.map_feature({
            features.equals_bin(1):'joint_lt_65_lt_65',
            features.equals_bin(2):'joint_lt_65_gt_65',
            features.equals_bin(3):'joint_gt_65_gt_65',
            features.equals_bin(4):'head',
            features.equals_bin(5):'single',
            features.equals_bin(6):'non_filer',
            }),'tax'),
        dff(features.map_feature({
            features.equals_bin(1):'yes',
            features.contains_bin([0,2]):'no',
        }),'union'),
        dff(features.map_feature({
            features.equals_bin(1):'private',
            features.equals_bin(2):'government',
            features.equals_bin(3):'self',
            features.equals_bin(4):'no_pay',
        }),'workclass'),
    ]
    remap_feats_pipeline = Pipeline([
        ('remap', sklearn_utils.transform_from_fxn(functools.partial(features.df_from_df_and_df_features, remap_feats))),
        ('drop_na', sklearn_utils.transform_from_fxn(lambda df: df.dropna(axis=0,how='any'))),
        ])
    remap_feats_data = remap_feats_pipeline.transform(data)
    #remap_feats_data = remap_feats_data.dropna(axis=0,how='any')
    print remap_feats_data.shape
    remap_feats_data.head(10)
    #remap_feats_data.apply(lambda col:col.isnull().sum(),axis=0)

    # for categorical features, bin them further to get their binary features
    cat_feats_bin_feats = [
        dff(features.bins_feature([features.equals_bin('citizen')], not_others=True),'citizenship'),
        dff(features.bins_feature([features.contains_bin(['bachelors','masters'])], not_others=True),'education'),
        dff(features.bins_feature([features.equals_bin('1000_None'),features.equals_bin('None_10')], not_others=True),'employer_size'),
        dff(features.bins_feature([features.equals_bin('full_time'),], not_others=True),'fulltime'),
        dff(features.bins_feature([features.equals_bin('householder'),], not_others=True),'household_summary'),
        dff(features.bins_feature([
            features.contains_bin(['agriculture','entertainment','finance_insurance','hospital']),
            features.not_bin(features.contains_bin(['manufacturing_durable','manufacturing_nondurable'])),
            features.contains_bin(['retail_trade','wholesale_trade']),
            features.contains_bin(['manufacturing_durable']),
            features.contains_bin(['manufacturing_nondurable']),
            features.contains_bin(['educational']),
            features.contains_bin(['hospital']),
            features.contains_bin(['medical']),
            features.contains_bin(['construction']),
            features.contains_bin(['business_repair','utilities']),
            features.contains_bin(['transportation']),
            features.contains_bin(['public_admin']),
            features.contains_bin(['finance_insurance']),
            features.contains_bin(['other_service','other_private_service','social_service','private_household_service']),
            features.contains_bin(['entertainment','communications']),
            features.contains_bin(['utilities','agriculture','mining','forestry_fisheries']),
            ], not_others=False),'industry'),
        dff(features.bins_feature([features.equals_bin('civilian_spouse'),], not_others=True),'marital'),
        dff(features.bins_feature([
            features.contains_bin(['executive','protective_service']),
            features.contains_bin(['farming_forestry','private_service','transportation']),
            features.contains_bin(['administrative']),
            features.contains_bin(['other_service','protective_service','private_service']),
            features.contains_bin(['skilled_tech_service']),
            features.contains_bin(['sales']),
            features.contains_bin(['factory']),
            features.contains_bin(['professional_specialty']),
            features.contains_bin(['handler_equip_cleaner']),
            features.contains_bin(['executive']),
            features.contains_bin(['transportation']),
            features.contains_bin(['technician','farming_forestry']),
            ], not_others=False),'occupation'),
        dff(features.bins_feature([features.equals_bin('white'),features.equals_bin('black')], not_others=True),'race'),
            dff(features.bins_feature([features.equals_bin('joint_lt_65_lt_65'),], not_others=True),'tax'),
            dff(features.bins_feature([features.equals_bin('no'),], not_others=True),'union'),
            dff(features.bins_feature([features.equals_bin('private'),], not_others=True),'workclass'),
#                       features.empirical_cat_df_feature('industry')
    ]
    cat_feats_bin_feats_pipeline = Pipeline([
        ('remap', sklearn_utils.transform_from_fxn(functools.partial(features.df_from_df_and_df_features, cat_feats_bin_feats))),
    ])
    cat_feats_bin_feats_data = cat_feats_bin_feats_pipeline.transform(remap_feats_data)
    print cat_feats_bin_feats_data.shape
    #print cat_feats_bin_feats_data.head(10)

    # get gender series
    gender = remap_feats_data['gender']

    # extract relevant columns for scalar features
    scalar_columns = [
        'age',
        'weeks_worked',
        'hourly_wage',
    ]
    scalar_feats_data = data[scalar_columns]

    # get binary features based on scalar data (filter it first)
    scalar_feats_bin_feats = [
        dff(features.bins_feature.from_boundaries([18,25,40,60,None],drop=False),'age'),
#               dff(features.bins_feature.from_boundaries([0,1,50,None],drop=False),'weeks_worked'),\
    ]
    scalar_feats_bin_feats_pipeline = Pipeline([
        ('worked_enough',sklearn_utils.transform_from_fxn(lambda df:df[df.weeks_worked>25.])),
        ('remap', sklearn_utils.transform_from_fxn(functools.partial(features.df_from_df_and_df_features, scalar_feats_bin_feats))),
        ])
    scalar_feats_bin_feats_data = scalar_feats_bin_feats_pipeline.transform(scalar_feats_data)
    print scalar_feats_bin_feats_data.shape
    #print scalar_feats_bin_feats_data.head(10)

    # get hourly wage
    wage = scalar_feats_data['hourly_wage']

    # get one big dataframe from which to extract raw input.  add column for indicator feature
    all_data = pd.concat([
        cat_feats_bin_feats_data,
        scalar_feats_bin_feats_data,
        pd.DataFrame({'indicator':np.ones(len(cat_feats_bin_feats_data))},index=cat_feats_bin_feats_data.index),
        pd.DataFrame({'hourly_wage':wage,'gender':gender}),
        ], join='inner', axis=1)
    print all_data.shape
    #print all_data.iloc[:,(-5):].head()

    x_ns = all_data.iloc[:,0:-2].values
    x_names = all_data.columns[0:-2]
    T_ns = (all_data.gender == 'M').values
    y_ns = all_data.hourly_wage.values

    return x_ns, x_names, T_ns.astype(int), y_ns
