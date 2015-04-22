import os
#home_folder = '/Users/fultonw/Documents/projects/code/data/problem_props_data'
home_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_folder = '%s/%s' % (home_folder, 'data')
parcels_file = '%s/%s' % (data_folder, 'parcels.csv')
