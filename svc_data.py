
import numpy as np


def load_raw_data():
    num_users = 40
    num_signatures = 40
    file_name_template = "svc/u{}s{}.txt"

    # signatures[user][g/f][sig][point][feature]
    signatures = []
    for u in range(num_users):
        genuine = []
        forgeries = []
        for s in range(num_signatures):
            signature = []
            file_name = file_name_template.format(u, s)
            with open(file_name) as file:
                for line in file.readlines():
                    labels = line.split()
                    signature.append([
                        float(labels[0]),  # x
                        float(labels[1]),  # y
                        float(labels[2]),  # timestamp
                        float(labels[6])   # pressure
                    ])
            if s < num_signatures/2:
                genuine.append(np.array(signature))
            else:
                forgeries.append(np.array(signature))
        signatures.append([
            genuine,
            forgeries
        ])

    return np.array(signatures)


def get_cleaned_data():
    data = load_raw_data()
    nu, ns, fs = data.shape

    # clean timestamps
    for u in range(nu):
        for s in range(ns):
            for f in range(fs):
                data[u, s, f][:, 2] -= min(data[u, s, f][:, 2])

    return data

