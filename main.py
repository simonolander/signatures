#!/bin/python
from math import sqrt, ceil

from dtw import dtw
from matplotlib import pyplot as plt
import storage
import hidden_signature

# Read data

data = storage.get_cleaned_data()

# Score data


def score(sig, test_sig):
    dist, cost, acc, path = dtw(sig, test_sig, hidden_signature.distance)
    # dist_cost = sum(cost[x, y] for (x, y) in zip(path[0], path[1]))
    return dist


def plot_sigs(*args):
    if len(args) > 1:
        n = int(ceil(sqrt(len(args))))
        f, axs = plt.subplots(ncols=n, nrows=n)
        axs.shape = (n * n)
    else:
        f, axs = plt.subplots(1)
        axs = [axs]

    for i, sig in enumerate(args):
        axs[i].plot(sig[:, 0], sig[:, 1])
    plt.tight_layout()
    plt.show()


##########

hids = storage.load_hidden_signatures()

for u in range(40):
    print sum([score(hids[u], data[u][0][i]) for i in range(20)])
    print sum([score(hids[u], data[u][1][i]) for i in range(20)])
    print
