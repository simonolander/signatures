#!/bin/python
from math import sqrt, ceil

from numpy import array, ones, inf, mean, std, interp, linspace, arange, zeros, average
from scipy.linalg import norm
from dtw import dtw
from matplotlib import pyplot as plt
import svc_data
import hidden_signature

# Read data

data = svc_data.get_cleaned_data()

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

user = 14
hidden_signatures = hidden_signature.estimate_hidden_signature(data[user][0])

for i, sig in enumerate(hidden_signatures):
    sg = sf = 0
    for g in data[user][0]:
        sg += score(sig, g)
    for f in data[user][1]:
        sf += score(sig, f)
    sg /= len(data[user][0])
    sf /= len(data[user][1])
    print "Hidden signature {}, genuine: {}".format(i, sg)
    print "Hidden signature {}, forgeries: {}".format(i, sf)

plot_sigs(*hidden_signatures)
