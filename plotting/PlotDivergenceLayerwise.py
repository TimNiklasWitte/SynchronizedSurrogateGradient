from LoadDataframe import *
from matplotlib import pyplot as plt
from plot_conf import *

import seaborn as sns

def main():

    num_layers = 4

    log_dir = PLOTTING_DIR +"/logs/"

    df = load_dataframe(log_dir)
 
    fig, axs = plt.subplots(nrows=1, ncols=num_layers, figsize=(12, num_layers))

    for layer_idx in range(num_layers):
        
        df_tmp = df.loc[:, [f"train divergence layer {layer_idx}", 
                                     f"test divergence layer {layer_idx}"]]
        df_tmp.columns = ["train", "test"]

        sns.lineplot(data=df_tmp.loc[:, ["train", "test"]], ax=axs[layer_idx])

        axs[layer_idx].set_title(f"layer {layer_idx}")


    for ax in axs:
        ax.grid()

    plt.tight_layout()
    plt.savefig(PLOTTING_DIR +"/plots/DivergenceLayerwise.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
