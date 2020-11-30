import csv
import numpy as np
import math

# reshape the last column to be 0 (no arrhythmia) or 1 (arrhythmia)
def collapse_label(x):
    # x = row vector
    if not math.isclose(x[-1], 1):
        x[-1] = 0
    else:
        x[-1] = 1
        # print(x)
    return x

# rescale each column according to range
# so a column with a wide range gets scaled down, and one
# with a small range (<1) gets scaled up
# This should improve the performance of some algorithms (eg kNN) significantly
# https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
#
def normalize_data(x):
    # x = column vector
    x_min = np.nanmin(x)
    x_range = np.nanmax(x) - x_min
    # don't divide by zero!
    if math.isclose(x_range, 0):
        #print('returning x, x_range:',x_range)
        return x
    # broadcast along each element of the column to account for None values
    return np.where(np.isnan(x), np.nan, (x - x_min) / (x_range))

def predict_missing_elements(data, k=5):
    import kNN
    # first, find the columns that have missing values
    isNaN = np.isnan(data)
    colHasNan = np.any(isNaN, axis=0)
    # print(colHasNan)

    # now, do column-wise interpretation

    model = kNN.kNN(k, method='regression')

    for i, col in enumerate(colHasNan):
        if (col):
            # print('predicting col', i, data[:,i], isNaN[:,i])

            # delete rows that have NaN in column i
            train = np.delete(data, isNaN[:,i], axis=0)

            # fit a kNN model to the data
            # delete column i - the column to predict
            model.fit(np.delete(train, i, axis=1), train[:,i])

            # this is inefficient but much easier to code
            # predict the value for all elements of column i
            p = model.predict(np.delete(data, i, axis=1))
            # if data[row,col] is None {data[row,col] = p} else {data[row,col] = data[row,col]}
            data[:,i] = np.where(isNaN[:,i], p, data[:,i])

            # print('predicted:', data[:,i])

    return data

def process_data(input_filename = 'arrhythmia.data', usecols=None,
        predict_missing=False, k_predict=5, collapse=True, normalize=False):

    data = np.genfromtxt(input_filename, usecols=usecols,
        delimiter=',',missing_values='?', filling_values=None, dtype=float)

    # should we have 16 classes (type of arrhythmia), or 1 (arrhythmia y/n)?
    if collapse:
        # print('before collapse:',data[-1])
        data = np.apply_along_axis(collapse_label, 1, data)
        # print('after collapse:',data[:,-1])

    # normalize the data
    if normalize:
        # this will affect the label if it hasn't been collapsed
        # not sure the best way to fix this -- right now just storing then
        # restoring the label after the apply
        # print('before norm:', data)
        label = data[:,-1]
        data = np.apply_along_axis(normalize_data, 0, data)
        data[:,-1] = label
        # print('after norm:', data)

    if predict_missing:
        #print('before prediction:', data)
        data = predict_missing_elements(data)
        #print('after prediction:', data)

    return data

def partition_data(data, partitions=[0.2,0.2,0.6]):
    data_partitions = [[] for _ in partitions]

    # right now: randomly split the data by first x rows, second y rows, etc
    m = np.shape(data)

    # number of rows in each partition
    if not math.isclose(sum(partitions), 1):
        print('error: partitions did not sum to 1! sum:', sum(partitions))
        return

    n_part = [p * m[0] for p in partitions]
    start = 0
    for i, p in enumerate(n_part):
        data_partitions[i] = data[math.floor(start):math.floor(start+p)]
        start += p

    return np.array(data_partitions, dtype=object)


if __name__=='__main__':
    # small test dataset
    d = process_data('small.data', normalize=True, predict_missing=True)
    print(d)