import os
from matplotlib import pyplot as plt
from plotting.plot_conf import PLOTTING_DIR

import numpy as np

def sigmoid(x, k):
    return 1 / (1 + np.exp(-k*x))

def sigmoid_prime(x, k):
    return k * np.exp(- k * x) / (1 + np.exp(-k*x))**2

def main():

    x = np.linspace(start=-4, stop=4, num=100)
    
    fig, axs = plt.subplots(nrows=1, ncols=2)
    for k in range(1, 8):
        y = sigmoid(x, k)
        axs[0].plot(x,y)

        y = sigmoid_prime(x,k)
        axs[1].plot(x,y, label=str(k))
    
    axs[0].set_xlabel("x")
    axs[1].set_xlabel("x")

    axs[0].set_ylabel("y")
    axs[1].set_ylabel("y")

    axs[0].set_title("y=sigmoid(x,k)")
    axs[1].set_title("y=sigmoid'(x,k)")
    
    axs[0].grid()
    axs[1].grid()

    axs[1].legend(title="k", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    os.makedirs( PLOTTING_DIR + "/plots", exist_ok=True)
    plt.savefig(PLOTTING_DIR +"/plots/Sigmoid_SigmoidPrime.png", dpi=200)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")