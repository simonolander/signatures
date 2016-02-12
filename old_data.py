from sklearn.preprocessing import MinMaxScaler
from numpy import array

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
