import os
home_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_folder = '%s/%s' % (home_folder, 'data')
file_path = '%s/%s' % (data_folder,'NSW.csv')
