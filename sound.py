#!/bin/python
from random import shuffle

import librosa
from numpy import array, ones, inf, mean, std
from scipy.linalg import norm
from dtw import dtw
from multiprocessing import Pool

with open("sounds/wavToTag.txt") as f:
    labels = array([line.replace("\n", "") for line in f])

mfccs = {}

labelToIndices = {}
for i, label in enumerate(labels):
    if label in labelToIndices:
        labelToIndices[label] += [i]
    else:
        labelToIndices[label] = [i]

for i in range(len(labels)):
    y, sr = librosa.load("sounds/{}.wav".format(i))
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)
    mfccs[i] = mfcc.T


def generate_train_test_set(p):
    train = []
    test = []

    for s in set(labels):
        data = labelToIndices[s]
        shuffle(data)
        train += data[:-p]
        test += data[-p:]

    return train, test


D = ones((len(labels), len(labels))) * -1


def dist(x, y):
    return norm(x - y, ord=1)


def cross_validation(train, test):
    score = 0.

    for i in test:
        x = mfccs[i]

        d_min, j_min = inf, -1
        for j in train:
            y = mfccs[j]

            d = D[i, j]
            if d == -1:
                d, _, _, _ = dtw(x, y, dist)
                D[i, j] = d

            if d < d_min:
                d_min = d
                j_min = j

        score += 1.0 if labels[i] == labels[j_min] else 0.0

    return score / len(test)


def f(p):
    return cross_validation(*generate_train_test_set(p))


P = array(range(1, 9))
print P
p = Pool(8)
rec = p.map(f, P)

print rec
