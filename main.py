#!/bin/python
from math import sqrt, ceil

from sklearn.preprocessing import MinMaxScaler
from numpy import array, ones, inf, mean, std, interp, linspace, arange, zeros, average
from scipy.linalg import norm
from dtw import dtw
from matplotlib import pyplot as plt

# Read data

signatures = []
labels = []

with open("data.txt", "r") as f:
    feature = []
    label = -1
    for line in f:
        if line == "\n":
            signatures.append(feature)
            labels.append(label)
            feature = []
            label = -1
            continue

        """ words: [x, y, time, tip, azimuth, altitude, pressure, forgery] """
        words = line.split()
        feature.append([
            float(words[0]),  # x
            float(words[1]),  # y
            float(words[2]),  # time
            float(words[6]),  # pressure
        ])
        label = int(words[7])  # forgery

signatures = array(signatures)
labels = array(labels)

# Normalize data [0., 1.]
transform = MinMaxScaler().fit_transform
signatures = map(transform, signatures)

# Split the data

genuine = [signatures[i] for i in range(0, len(signatures)) if labels[i] == 0]
forgeries = [signatures[i] for i in range(0, len(signatures)) if labels[i] == 1]

train_percent = 0.5
train_num_genuine = int(train_percent * len(genuine))
train_num_forgeries = int(train_percent * len(forgeries))

train_genuine = genuine[:train_num_genuine]
test_genuine = genuine[train_num_genuine:]
train_forgeries = forgeries[:train_num_forgeries]
test_forgeries = forgeries[train_num_forgeries:]

train_features = train_genuine + train_forgeries
train_labels = [0] * train_num_genuine + [1] * train_num_forgeries
test_features = test_genuine + test_forgeries
test_labels = [0] * len(test_genuine) + [1] * len(test_forgeries)

print "Num train:", len(train_features)
print "Num test:", len(test_features)

for train in train_features:
    for test in test_features:
        assert train != test, "Train equals test"

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
        axs.shape = (n*n)
    else:
        f, axs = plt.subplots(1)
        axs = [axs]

    for i, sig in enumerate(args):
        axs[i].plot(sig[:, 0], sig[:, 1])
    plt.tight_layout()
    plt.show()


sigs = array(genuine)

sigs = estimate_hidden_signature(sigs)

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
