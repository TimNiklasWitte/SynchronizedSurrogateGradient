from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():

    num_layers = 4

    log_dir = "./../baseline/snn/logs"

    df = load_dataframe(log_dir)
 
    fig, axs = plt.subplots(nrows=1, ncols=num_layers + 1, figsize=(16, num_layers + 1))

    for layer_idx in range(num_layers):
        

        sns.lineplot(data=df.loc[:, [f"test divergence layer {layer_idx}"]], ax=axs[layer_idx], legend=None)

        axs[layer_idx].set_title(f"layer {layer_idx}")


    sns.lineplot(data=df.loc[:, ["test divergence"]], ax=axs[4], legend=None)
    axs[4].set_title(f"average")

    for ax in axs:
        ax.grid()
        ax.set_ylabel("Activation divergence")

    plt.tight_layout()
    plt.savefig("./plots/DivergenceLayerwise_SNN.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
