from scipy.linalg import norm
from numpy import zeros, linspace, arange, interp, average, array
from dtw import dtw


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


def estimate_hidden_signature(signatures, iterations=4):
    length = sum([len(x) for x in signatures]) / len(signatures)
    normalized_signatures = normalize_spaces(signatures, length)
    hidden = average(normalized_signatures, axis=0)

    hiddens = [hidden]

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
        hiddens.append(hidden)

    return hiddens