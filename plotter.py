#!/usr/bin/python

import sys
from glob import glob

import matplotlib.pyplot

import numpy

if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "-r"):
    print "Usage:",
    print __file__, "file1 [file2] ..."
    sys.exit(1)

else:
    file_names = sys.argv[1:]

for file_name in file_names:
    signature = []
    with open(file_name) as file:
                for line in file.readlines():
                    labels = line.split()
                    signature.append([
                        float(labels[0]),  # x
                        float(labels[1]),  # y
                        float(labels[2]),  # timestamp
                        float(labels[6])   # pressure
                    ])
    signature = numpy.array(signature)
    matplotlib.pyplot.plot(signature[:, 0], signature[:, 1])
    print "Showing file: ", file_name
    matplotlib.pyplot.show()
