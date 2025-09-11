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

    path = f"./../baseline/ann/logs"
    df = load_dataframe(path)

    axs[0].plot(epochs, np.array(df.loc[:, ["test accuracy"]]), label="ANN", color="green")


    #
    # SNN
    epochs = np.arange(33)

    path = f"./../baseline/snn/logs"
    df = load_dataframe(path)

    axs[0].plot(epochs, np.array(df.loc[:, ["test accuracy"]]), label="SNN")
    axs[0].set_title("Test accuracy")

    axs[1].plot(epochs, np.array(df.loc[:, ["test divergence"]]), label="SNN")
    axs[1].set_title("Test divergence")


    #
    # SynSurrogateGrad: SNN
    #

    path = f"./../SynSurrogateGrad/snn/logs"
    df = load_dataframe(path)


    axs[0].plot(epochs, np.array(df.loc[:, ["test accuracy"]]), label="SynSurrogateGrad: SNN")
 

    axs[1].plot(epochs, np.array(df.loc[:, ["test divergence"]]), label="SynSurrogateGrad: SNN")

    

    axs[0].legend()


    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Activation divergence")

    for ax in axs:
        ax.grid()
        ax.set_xlabel("Epoch")

    plt.tight_layout()

    plt.savefig("./plots/CompareTrainingFFN.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
