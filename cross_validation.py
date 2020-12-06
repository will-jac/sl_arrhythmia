import numpy as np
import preprocess
import risk

def cross_validation(X, y, models):
    k = len(models)

    partitioned_data = preprocess.partition_data(X, y, partitions=[1/k for _ in range(k)])

    # print(partitioned_data)

    r = []

    for i, model in enumerate(models):
        valid = partitioned_data[i]
        train_X = []
        train_y = []
        primed = False
        for j in range(k):
            if i == j:
                continue
            if not primed:
                train_X = partitioned_data[j][0]
                train_y = partitioned_data[j][1]
                primed = True
            else:
                train_X = np.append(train_X, partitioned_data[j][0], axis=0)
                train_y = np.append(train_y, partitioned_data[j][1], axis=0)

        model.fit(train_X, train_y)

        p = model.predict(valid[0])

        r.append(risk.empirical_risk('mse', p, valid[1]))

    return r
