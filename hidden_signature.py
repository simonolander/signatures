from scipy.linalg import norm
from numpy import zeros, linspace, arange, interp, average, array
from dtw import dtw
from multiprocessing import Pool


def distance(x, y):
    """
    A wrapper for scipy.linalg.norm(x - y, ord=1)
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
    n_sig = zeros(shape=(length, len(sig[0])))
    new_xs = linspace(0, len(sig) - 1, length)
    old_xs = arange(0, len(sig))
    for i in range(len(sig.T)):
        n_sig.T[i] = interp(new_xs, old_xs, sig.T[i])
    return n_sig


def normalize_spaces(sigs, length):
    """
    Creates a numpy array with the normalization of all the signatures in sigs.
    :param sigs:
    :param length:
    :return:
    """
    return array([normalize_space(sig, length) for sig in sigs])


def estimate_hidden_signature(signatures, iterations=4):
    """
    Creates an average "hidden" signature the signatures in signatures.
    The signatures are all supposed to come from the same author.

    :param signatures: The set of signatures from some author
    :param iterations: The number of iterations for improving the initial estimate
    :return: The average "hidden" signature
    """
    length = sum([len(x) for x in signatures]) / len(signatures)
    normalized_signatures = normalize_spaces(signatures, length)
    hidden = average(normalized_signatures, axis=0)

    for k in range(iterations):
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

    return hidden


def estimate_hidden_signatures(user_signatures):
    """
    :param user_signatures: array (n_users, n_genuine_signatures, n_samples (varied), n_features)
    :return:
    """
    pool = Pool()
    hidden_signatures = pool.map(estimate_hidden_signature, user_signatures)
    return hidden_signatures
