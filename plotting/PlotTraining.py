from plotting.plot_conf import PLOTTING_DIR, PLOT_DIVERGENCE
from plotting.LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = PLOTTING_DIR + "/logs/"

    df = load_dataframe(log_dir)

    ncols = 3 if PLOT_DIVERGENCE else 2

    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 4))

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
    if PLOT_DIVERGENCE:
        df_tmp = df.loc[:, ["train divergence", "test divergence"]]
        df_tmp.columns = ["train", "test"]
        sns.lineplot(data=df_tmp.loc[:, ["train", "test"]], ax=axs[2])
        axs[2].set_title("Divergence")

    for ax in axs:
        ax.grid()

    plt.tight_layout()
    os.makedirs( PLOTTING_DIR + "/plots", exist_ok=True)
    plt.savefig( PLOTTING_DIR + "/plots/Training.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
