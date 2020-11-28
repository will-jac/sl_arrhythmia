import numpy as np
import preprocess
import risk

def cross_validation(data, models):
    k = len(models)

    partitioned_data = preprocess.partition_data(data, partitions=[1/k for _ in range(k)])

    r = []

    for i, model in enumerate(models):
        valid = partitioned_data[i]

        hole_data = np.delete(partitioned_data, i, 0)
        train = hole_data[0]
        for i in range(1, len(hole_data)):
            np.append(train, partitioned_data[i])

        # print(train)
        model.fit(train[:,0:-1], train[:,-1])
        # print(valid)
        p = model.predict(valid[:,0:-1])

        r.append(risk.empirical_risk('mse', p, valid[:,-1]))

    return r
