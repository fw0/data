import os
home_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_folder = '%s/%s' % (home_folder, 'data')
sexual_function_data_file = '%s/%s' % (data_folder, 'sex.csv')
xs_file = '%s/%s' % (data_folder, 'xs.csv')
old_good_sexual_function_data_pickle_file = '%s/%s' % (data_folder,'old_good_sexual_function_data.pickle')
from_ec2_pids = '%s/%s' % (data_folder,'pids_for_paper')
