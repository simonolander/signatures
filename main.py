#!/bin/python
from math import sqrt, ceil

from numpy import array, ones, inf, mean, std, interp, linspace, arange, zeros, average
from scipy.linalg import norm
from dtw import dtw
from matplotlib import pyplot as plt
import svc_data

# Read data

data = svc_data.get_cleaned_data()



# Score data


def distance(x, y):
    return norm(x - y, ord=1)


def normalize_space(sig, length):
    n_sig = zeros(shape=(length, len(sig[0])))
    new_xs = linspace(0, len(sig) - 1, length)
    old_xs = arange(0, len(sig))
    for i in range(len(sig.T)):
        n_sig.T[i] = interp(new_xs, old_xs, sig.T[i])
    return n_sig


def normalize_spaces(sigs, length):
    return array([normalize_space(sig, length) for sig in sigs])


def estimate_hidden_signature(signatures):
    length = sum([len(x) for x in signatures]) / len(signatures)
    normalized_signatures = normalize_spaces(signatures, length)
    hidden = average(normalized_signatures, axis=0)

    hiddens = [hidden]

    for k in range(4):
        for n, sig in enumerate(signatures):
            _, _, _, path = dtw(hidden, sig, distance)
            sig_prime = normalized_signatures[n]
            sig_prime.fill(0)
            num_x = zeros(length)

            for (x, y) in zip(*path):
                sig_prime[x] += sig[y]
                num_x[x] += 1

            num_x.shape = (length, 1)
            sig_prime /= num_x

        hidden = average(normalized_signatures, axis=0)
        hiddens.append(hidden)

    return hiddens


def predict(features_train, labels_train, features_test):
    dist_mat = ones((len(labels), len(labels))) * -1
    predicted_labels = []

    for i, x in enumerate(features_test):
        d_min, j_min = inf, -1
        for j, y in enumerate(features_train):

            d = dist_mat[i, j]
            if d == -1:
                d, _, _, _ = dtw(x, y, distance)
                dist_mat[i, j] = d

            if d < d_min:
                d_min = d
                j_min = j

        predicted_labels += [labels_train[j_min]]

    return predicted_labels


def score(sig, test_sig):
    dist, cost, acc, path = dtw(sig, test_sig, distance)
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

# x = sigs[:, 0]
# y = sigs[:, 1]
# plt.plot(x, y)
# plt.show()

for i, sig in enumerate(sigs):
    sg = sf = 0
    for g in genuine:
        sg += score(sig, g)
    for f in forgeries:
        sf += score(sig, f)
    sg /= len(genuine)
    sf /= len(forgeries)
    print "Hidden signature {}, genuine: {}".format(i, sg)
    print "Hidden signature {}, forgeries: {}".format(i, sf)

plot_sigs(*sigs)
