from scipy.linalg import norm
import numpy as np
from dtw import dtw
import multiprocessing as mp
import math
from matplotlib import pyplot as plt
import logging


def distance(x, y):
    """
    A wrapper for scipy.linalg.norm(x - y, ord=1)
    :param x:
    :param y:
    """
    return norm(x - y, ord=1)


def normalize_space(sig, length):
    """
    Takes a signature of shape (n, f), where n is the number of samples and f is the number of features,
    and produces an interpolated signature of shape (length, f), where length is the provided length, and f
    stays the same.

    :param sig: The signature, shape (n, f)
    :param length: The length the signature should be normalized to
    :return: The normalized signature, shape (length, f)
    """
    n_sig = np.zeros(shape=(length, len(sig[0])))
    new_xs = np.linspace(0, len(sig) - 1, length)
    old_xs = np.arange(0, len(sig))
    for i in range(len(sig.T)):
        n_sig.T[i] = np.interp(new_xs, old_xs, sig.T[i])
    return n_sig


def normalize_spaces(sigs, length):
    """
    Creates a numpy array with the normalization of all the signatures in sigs.
    :param sigs:
    :param length:
    :return:
    """
    return np.array([normalize_space(sig, length) for sig in sigs])


def compute_error_vector(hidden_signature, signatures):
    """
    Returns the expected deviation from the hidden signature for each feature in each point.
    :param np.ndarray hidden_signature: The hidden signature
    :param np.ndarray signatures: The set of normalized signatures to compute the error for
    :return: The root mean square error
    """
    diffs = signatures - hidden_signature
    diff_squares = diffs**2
    sums = np.sum(diff_squares, axis=0)
    means = sums / float(len(signatures))
    root_mean_square_error = np.sqrt(means)

    return root_mean_square_error


def iterate_hidden_signature_estimate(hidden_signature, signature):
    n_samples, _ = hidden_signature.shape
    _, _, _, path = dtw(hidden_signature, signature, distance)
    sig_prime = np.zeros(shape=hidden_signature.shape)
    num_x = np.zeros(shape=(n_samples, 1))

    for (x, y) in zip(*path):
        sig_prime[x] += signature[y]
        num_x[x] += 1.

    sig_prime /= num_x
    return sig_prime


def iterate_hidden_signature_estimate_star(args):
    return iterate_hidden_signature_estimate(*args)


def estimate_hidden_signature(signatures, max_iterations=1000, epsilon=0.001, n_jobs=1):
    """
    Creates an average "hidden" signature the signatures in signatures.
    The signatures are all supposed to come from the same author.

    :param signatures: The set of signatures from some author
    :param max_iterations: The maximum number of iterations for improving the initial estimate
    :param epsilon: Break the iterative improvement of the signatures if the difference in score is less than epsilon
    :param int n_jobs: The number of cores the process can use, -1 for all cores
    :return: The average "hidden" signature
    """
    n_cores = int(n_jobs)
    if n_cores == -1:
        n_cores = None

    length = sum([len(x) for x in signatures]) / len(signatures)
    normalized_signatures = normalize_spaces(signatures, length)
    hidden = np.average(normalized_signatures, axis=0)
    old_score = dtw_avg_distance(hidden, normalized_signatures)

    break_step = -1

    pool = mp.Pool()

    for k in range(max_iterations):
        normalized_signatures = pool.map(
            iterate_hidden_signature_estimate_star,
            [(hidden, sig) for sig in normalized_signatures]
        )
        hidden = np.average(normalized_signatures, axis=0)
        new_score = dtw_avg_distance(hidden, normalized_signatures)

        break_step = k
        if abs(new_score - old_score) < epsilon:
            break

    error = compute_error_vector(hidden, normalized_signatures)
    logging.debug("Hidden signature estimated, iteration step:", break_step)

    return hidden, error


def estimate_hidden_signatures(user_signatures):
    """
    :param user_signatures: array (n_users, n_genuine_signatures, n_samples (varied), n_features)
    :return:
    """
    pool = mp.Pool()
    hidden_signatures = pool.map(estimate_hidden_signature, user_signatures)
    return hidden_signatures


def dtw_avg_distance(sig, test_sigs):
    score = 0
    for test_sig in test_sigs:
        dist, cost, acc, path = dtw(sig, test_sig, distance)
        score += dist

    return score / len(test_sigs)


def score_rmse(hidden_signature, rmse, signature):
    if not len(signature) == len(hidden_signature):
        signature = normalize_space(signature, len(hidden_signature))

    difference = signature - hidden_signature
    difference /= rmse

    variance = np.var(difference, axis=0)
    euclidean_norm = np.linalg.norm(variance)

    return euclidean_norm


def plot_signatures(*args):
    if len(args) > 1:
        n = int(math.ceil(math.sqrt(len(args))))
        f, axs = plt.subplots(ncols=n, nrows=n)
        axs.shape = (n * n)
    else:
        f, axs = plt.subplots(1)
        axs = [axs]

    for i, sig in enumerate(args):
        axs[i].plot(sig[:, 0], sig[:, 1])
    plt.tight_layout()
    plt.show()
