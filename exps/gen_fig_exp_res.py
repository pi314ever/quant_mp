import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

# file_name = 'exps/results/qat_float_4_None_ResNet.pickle'
file_name = "exps/results/qat_float_4_None_e2m1_ResNet.pickle"
with open(file_name, "rb") as handle:
    return_dict = pickle.load(handle)


def moving_average(a, n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# figure, (ax1, ax2) = plt.subplots(2, 1)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
for method in return_dict:
    ax1.plot(moving_average(return_dict[method][0], n=5), label=method)
    ax1.legend()
    ax1.set_title("Train loss")
    ax1.set_yscale("log")
    ax1.grid(True)

    ax2.plot(moving_average(return_dict[method][1], n=15), label=method)
    ax2.legend()
    ax2.set_title("Test loss")
    ax2.grid(True)


fig1.savefig(file_name.split(".")[0] + "_train.jpg")
fig1.show()

fig2.savefig(file_name.split(".")[0] + "_test.jpg")
fig2.show()
