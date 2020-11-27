import numpy as np
import preprocess

def mse(predict, true):
    return np.sum(np.square(predict - true))

def cross_validation(models, collapse=True):
    k = len(models)

    data = preprocess.process_data(partitions=[1/k for _ in range(k)], collapse=collapse)

    risk = []

    for i, model in enumerate(models):
        valid = data[i]

        hole_data = np.delete(data, i, 0)
        train = hole_data[0]
        for i in range(1, len(hole_data)):
            np.append(train, data[i])

        # part, rows, cols = train.shape
        # rows = part*rows
        # np.reshape(train, newshape=(rows, cols))
        # print(train)
        model.fit(train[:,0:-1], train[:,-1])
        # print(valid)
        p = model.predict(valid[:,0:-1])

        risk.append(mse(p, valid[:,-1]))

    return risk
