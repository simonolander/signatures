#!/bin/python

import sys
import glob
import matplotlib.pyplot

import numpy

if len(sys.argv) == 1:
    print "Usage:", __file__, "file1 [file2] ..."
    sys.exit(1)

for file_name in sys.argv[1:]:
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
