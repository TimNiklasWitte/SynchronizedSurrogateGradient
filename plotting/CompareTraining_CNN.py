#from plotting.plot_conf import PLOTTING_DIR, PLOT_DIVERGENCE
from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np
def main():

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    #
    # ANN
    #

    epochs = np.arange(32)

    path = f"./../baseline/cnn/logs"
    df = load_dataframe(path)
 
    axs[0].plot(epochs, np.array(df.loc[:, ["test accuracy"]]).reshape(-1), label="CNN", color="green")


    #
    # SNN
    epochs = np.arange(32)

    path = f"./../baseline/scnn/logs"
    df = load_dataframe(path)

    axs[0].plot(epochs, np.array(df.loc[:, ["test accuracy"]]).reshape(-1), label="SCNN")
    axs[0].set_title("Test accuracy")

    axs[1].plot(epochs, np.array(df.loc[:, ["test divergence"]]).reshape(-1), label="SCNN")
    axs[1].set_title("Test divergence")


    #
    # SynSurrogateGrad: SNN
    #

    epochs = np.arange(33)
    path = f"./../SynSurrogateGrad/scnn/logs"
    df = load_dataframe(path)


    axs[0].plot(epochs, np.array(df.loc[:, ["test accuracy"]]), label="SynSurrogateGrad: SCNN")
 

    axs[1].plot(epochs, np.array(df.loc[:, ["test divergence"]]), label="SynSurrogateGrad: SCNN")

    

    axs[0].legend()


    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Activation divergence")

    for ax in axs:
        ax.grid()
        ax.set_xlabel("Epoch")

    plt.tight_layout()

    plt.savefig("./plots/CompareTraining_CNN.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
