import pandas as pd
import numpy as np
import os

home_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_folder = '%s/%s' % (home_folder, 'data')

def raw_violent_data():
    """
    returns feature df, y series, subgroup indicator df
    """

    file_name = 'crime_violent_processed.csv'

    df = pd.read_csv('%s/%s' % (data_folder, file_name))
    
    binary_cols = [
        'Male',
        'Hispanic',
        'Black',
        'OtherRace',
        'AlcoholOrDrugDependency',
        'MentalHealthDiagnosis',
        'HasChildren',
        'Abuse',
        'Neglect',
        'SexualAbuse',
        'Del1W1',
        'Del2W1',
        'Del3W1',
        'Del4W1',
        'Del5W1',
        'Del6W1',
        'Del7W1',
        'Del8W1',
        'Del9W1',
        'Del10W1',
        'DamagedPropW1',
        'AllTheftW1',
        'SmallTheftW1',
        'LargeTheftW1',
        'EntryTheftW1',
        'SellDrugsW1',
        'ViolentCrimeW1',
        'ViolentCrimeWithGroupW1',
        'AnyDelW1',
        'AnyArrestW1',
        'EntryOverOver12',
        'TotalPlacementsOver5W1',
        'FosterCare',
        'KinCare',
        'GroupCare',
        'IndothCare',
        'NotCloseToMom',
        'MissingMom',
        'NotCloseToDad',
        'MIssingDad',
        'MissingMomAndDad',
        'InSchool',
        'CollegePlans',
        'Employed',
        ]
        
    scalar_cols = [
        'SumOfAbuseItems',
        'SumOfNeglectItems',
        'SumOfILSServices',
        'MomClosenessScore',
        'DadClosenessScore',
        'CloseToCaregiver',
        'SupportScore',
        'HelpScore',
        ]
        
    y_col = 'Class'

    z_cols = binary_cols
    
    all_cols = binary_cols + scalar_cols + [y_col] + z_cols
    df = df[all_cols]
    df = df.dropna(axis=0)

    binary_df = df[binary_cols]
    scalar_df = df[scalar_cols]
    ys = df[y_col]
    z_df = df[z_cols]

    scalar_df = (scalar_df - scalar_df.mean(axis=0)) / scalar_df.std(axis=0)

    feat_df = pd.concat((scalar_df, binary_df), axis=1)

    ys = (2 * ys) - 1.

    return feat_df, ys, z_df
