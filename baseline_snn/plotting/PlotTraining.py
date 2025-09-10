from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = "../logs/"

    df = load_dataframe(log_dir)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    #
    # Loss
    #

    df_tmp = df.loc[:, ["train loss", "test loss"]]
    df_tmp.columns = ["train", "test"]
    sns.lineplot(data=df_tmp.loc[:, ["train", "test"]], ax=axs[0])
    axs[0].set_title("Loss")

    #
    # Accuracy
    #

    df_tmp = df.loc[:, ["train accuracy", "test accuracy"]]
    df_tmp.columns = ["train", "test"]
    sns.lineplot(data=df_tmp.loc[:, ["train", "test"]], ax=axs[1])
    axs[1].set_title("Accuracy")

    #
    # Divergence
    #

    df_tmp = df.loc[:, ["train divergence", "test divergence"]]
    df_tmp.columns = ["train", "test"]
    sns.lineplot(data=df_tmp.loc[:, ["train", "test"]], ax=axs[2])
    axs[2].set_title("Divergence")

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
