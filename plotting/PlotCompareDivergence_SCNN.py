from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():

    num_layers = 3

    log_dir = "./../baseline/scnn/logs"
    df_snn = load_dataframe(log_dir)
    df_snn["type"] = "SCNN"


    log_dir = "./../SynSurrogateGrad/scnn/logs"
    df_synSurrogateGrad_snn = load_dataframe(log_dir)
    df_synSurrogateGrad_snn["type"] = "SCNN: SynSurrogateGrad"

    df = pd.concat([df_snn, df_synSurrogateGrad_snn])


    fig, axs = plt.subplots(nrows=1, ncols=num_layers + 1, figsize=(16, num_layers + 1))

    for layer_idx in range(num_layers):
        

        sns.lineplot(x="Epoch", y=f"test divergence layer {layer_idx}", hue="type", data=df.loc[:, [f"test divergence layer {layer_idx}", "type"]], ax=axs[layer_idx], legend=None)

        axs[layer_idx].set_title(f"layer {layer_idx}")


    sns.lineplot(x="Epoch", y="test divergence", hue="type", data=df.loc[:, ["test divergence", "type"]], ax=axs[3])
    axs[3].set_title(f"average")

    for ax in axs:
        ax.grid()
        ax.set_ylabel("Activation divergence")


    
    # Extract handles and labels for global legend
    handles, labels = axs[3].get_legend_handles_labels()
    axs[3].get_legend().remove()

    # Place one legend below all plots
    fig.legend(
        handles, labels, 
        loc="lower center", 
        ncol=len(labels), 
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig("./plots/CompareDivergence_SCNN.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
