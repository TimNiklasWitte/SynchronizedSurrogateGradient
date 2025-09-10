from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = "../logs/"

    df = load_dataframe(log_dir)
    print(df.columns)
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    sns.lineplot(data=df.loc[:, ["train loss", "test loss"]], ax=axs[0])


    sns.lineplot(data=df.loc[:, ["train accuracy", "test accuracy"]], ax=axs[1])

    sns.lineplot(data=df.loc[:, ["train divergence", "test divergence"]], ax=axs[2])

    for ax in axs:
        ax.grid()

    plt.tight_layout()
    plt.savefig("./plots/Training.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
