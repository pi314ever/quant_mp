import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np


def moving_average(a, n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QAT experiment results")
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=["fp", "int"],
        help="dtype to plot results for: 'fp' or 'int'",
    )
    args = parser.parse_args()

    if args.dtype == "int":
        file_name = "exps/results/qat_int4_None_ResNet.pickle"
    elif args.dtype == "fp":
        file_name = "exps/results/qat_fp4_e2m1_None_ResNet.pickle"
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Expected 'fp' or 'int'.")

    with open(file_name, "rb") as handle:
        return_dict = pickle.load(handle)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)

    for data_type, alg in return_dict:
        label = data_type
        if alg:
            label += "-" + alg

        ax1.plot(moving_average(return_dict[(data_type, alg)][0], n=5), label=label)
        ax2.plot(moving_average(return_dict[(data_type, alg)][1], n=15), label=label)

    ax1.legend()
    ax1.set_yscale("log")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epochs")
    ax1.grid(True)

    ax2.legend()
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epochs")
    ax2.grid(True)

    base = file_name.rsplit(".", 1)[0]
    fig1.savefig(base + "_train.pdf", format="pdf", dpi=300)
    fig2.savefig(base + "_test.pdf", format="pdf", dpi=300)
