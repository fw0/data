import constants
import pandas as pd

def NSW_data():
    return pd.DataFrame.from_csv(constants.file_path)
