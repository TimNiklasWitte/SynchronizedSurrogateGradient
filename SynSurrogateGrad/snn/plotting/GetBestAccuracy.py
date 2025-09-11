from LoadDataframe import *

import seaborn as sns
import numpy as np

def main():
    log_dir = "../logs/"

    df = load_dataframe(log_dir)

    df = df.loc[:, ["test accuracy"]]
    
    print(np.max(df))
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
