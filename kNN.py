
import math
import numpy as np

# kNN

def distance(x1, x2):
    # right now using L2 (Euclidean) norm
    return(math.sqrt(np.sum(np.square(x1 - x2))))

def mean_squared_error_loss(predict, true):
    return np.square(predict - true)

def zero_one_loss(predict, true):

    def eq(v):
        if (v != 0):
            return 0
        return 1

    return np.apply_along_axis(eq, 1, predict - true)

def empirical_risk(loss, predict, true):
    return np.mean(loss(predict, true))

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

    def predict(self, data):

        p = np.zeros(data.shape[0])
        # print(data.shape)

        def prediction(row, k, features, labels):
            # print(k)
            # print(features)
            # print(labels)
            # find the k nearest data element
            k_closest = [None for _ in range(k)]
            for i, f in enumerate(features):
                d = distance(row, f)
                if k_closest[-1] is None or k_closest[-1][0] > d:
                    # the loop is here to speed up computation (most of the above will be false, I think)
                    for j in range(k):
                        if k_closest[j] is None or k_closest[j][0] > d:
                            k_closest[j] = [d, labels[i]]
                            # don't fill up k_closest!
                            break
            # now, do the classification
            # could do weighting here
            unique, counts = np.unique([k[1] for k in k_closest], return_counts=True)
            return unique[np.argmax(counts)]

        return np.apply_along_axis(prediction, 1, data, self.k, self.features, self.labels)

if __name__=='__main__':
    import kFold

    # data = [[testing], [validation], [training]]
    # data = preprocess.process_data(partitions=[0.2,0.8])#[0.1 for i in range(10)])

    models = [kNN(i) for i in range(1,10)]
    risk = kFold.cross_validation(models)
    print(risk)
    # print(data[1][:,0:-1])
    # print(data[1][:,-1])
    # model.fit(data[1][:,0:-1], data[1][:,-1])
    # predictions = model.predict(data[0][:,0:-1])
    # # print(data[0][:,0:-1])
    # # print(data[0][:,-1])
    # print(predictions)
    # loss = np.mean(mean_squared_error_loss(predictions, data[0][:,-1]))
    # print(loss)