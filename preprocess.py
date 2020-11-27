import csv
import numpy as np
import math

# reshape the last column to be 0 (no arrhythmia) or 1 (arrhythmia)
def condense(x):
    # print(x)
    if not math.isclose(x[-1], 1):
        x[-1] = 0
    else:
        x[-1] = 1
        # print(x)
    return x

def process_data(input_filename = 'arrhythmia.data', partitions=[0.2,0.2,0.6], collapse=True):
    data_partitions = [[] for _ in partitions]

    # https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt
    data = np.genfromtxt(input_filename, delimiter=',',missing_values='?', filling_values=None, )

    # right now: randomly split the data by first x rows, second y rows, etc
    m = np.shape(data)

    if collapse:
        np.apply_along_axis(condense, 1, data)

    # number of rows in each partition
    if not math.isclose(sum(partitions), 1):
        print('error: partitions did not sum to 1! sum:', sum(partitions))
        return

    n_part = [p * m[0] for p in partitions]
    start = 0
    for i, p in enumerate(n_part):
        data_partitions[i] = data[math.floor(start):math.floor(start+p)]
        start += p

    return np.array(data_partitions)

if __name__=='__main__':
    d = process_data()
    print(d)