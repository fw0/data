import pandas as pd
import constants
import pdb

def salary_df():
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
    return pd.DataFrame(d).T
