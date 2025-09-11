from tbparse import SummaryReader
import pandas as pd

import os

def load_dataframe(log_dir):

    

    dir_list = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    
    df_list = []
    for dir_name in dir_list:
        
        reader = SummaryReader(f"{log_dir}/{dir_name}")

        df = reader.scalars

        # {Accuracy,Loss}_{Test,Train}
        dir_name_splitted = dir_name.split("_")
        
        if len(dir_name_splitted) == 4:
            column_name = f"{dir_name_splitted[-1]} {dir_name_splitted[0]} {dir_name_splitted[1]} {dir_name_splitted[2]}".lower()
        else:
            column_name = f"{dir_name_splitted[1]} {dir_name_splitted[0]}".lower()

        df = df.rename(columns={'step': 'Epoch', "value": column_name})

        df = df.set_index(['Epoch'])

        df = df.drop("tag", axis=1)

        df_list.append(df)
  
    df = pd.concat(df_list, axis=1)

    return df