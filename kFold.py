import numpy as np
import preprocess
import risk

def cross_validation(models, collapse=True):
    k = len(models)

    data = preprocess.process_data(partitions=[1/k for _ in range(k)], collapse=collapse)

    r = []

    for i, model in enumerate(models):
        valid = data[i]

        hole_data = np.delete(data, i, 0)
        train = hole_data[0]
        for i in range(1, len(hole_data)):
            np.append(train, data[i])

        # print(train)
        model.fit(train[:,0:-1], train[:,-1])
        # print(valid)
        p = model.predict(valid[:,0:-1])

        r.append(risk.empirical_risk('mse', p, valid[:,-1]))

    return r
