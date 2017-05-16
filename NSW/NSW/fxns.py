import constants
import pandas as pd

def NSW_data():
    ans = pd.DataFrame.from_csv(constants.file_path)
    ans['white'] = map(int,(ans.black + ans.hispan == 0))
    return ans
