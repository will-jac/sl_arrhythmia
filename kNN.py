
import math
import numpy as np

# kNN

def distance(x1, x2):
    # right now using L2 (Euclidean) norm
    return(math.sqrt(np.sum(np.square(x1 - x2))))

class kNN:

    def __init__(self, k):
        # TODO: some metaparameters, etc
        self.k = k

    def fit(self, features, labels):
        # for classification
        # predict argmax (sum (w_i * 1[f(w_i) = y]))
        # w_i = 1/(d(x_q, x_i)^2)
        self.features = features
        self.labels = labels

    def prediction(self, row):
        # find the k nearest data element
        k_closest = [None for _ in range(self.k)]
        for i, f in enumerate(self.features):
            d = distance(row, f)
            if k_closest[-1] is None or k_closest[-1][0] > d:
                # the loop is here to speed up computation (most of the above will be false, I think)
                for j in range(self.k):
                    if k_closest[j] is None or k_closest[j][0] > d:
                        k_closest[j] = [d, self.labels[i]]
                        # don't fill up k_closest!
                        break
        # now, do the classification
        # could do weighting here
        unique, counts = np.unique([k[1] for k in k_closest], return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, data):
        p = np.zeros(data.shape[0])
        # print(data.shape)
        return np.apply_along_axis(self.prediction, 1, data)

if __name__=='__main__':
    import kFold
    import preprocess
    import risk

    models = [kNN(i) for i in range(1,10)]
    r = kFold.cross_validation(models)

    print(r)
    k = np.argmax(r)

    data = preprocess.process_data(partitions=[0.2,0.8])
    train = data[1]
    valid = data[0]

    model = kNN(k)

    model.fit(train[:,0:-1], train[:,-1])
    p = model.predict(valid[:,0:-1])

    print(risk.empirical_risk('mse', p, valid[:,-1]))