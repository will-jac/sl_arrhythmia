
import math
import numpy as np

# kNN

def distance(x1, x2):
    # treat NAN as 0 (eg no impact on the distance)
    # need to cast dtype to a float because it's set to object by the partitioning
    # x1 = np.array(x1, dtype=float)
    # x2 = np.array(x1, dtype=float)

    x1 = np.where(np.isnan(x1), 0, x1)
    x2 = np.where(np.isnan(x2), 0, x2)
    # right now using L2 (Euclidean) norm
    return(math.sqrt(np.sum(np.square(x1 - x2))))

class kNN:

    def __init__(self, k, method='classification', distance='euclidean'):
        # TODO: additional metaparameters such as lp norms, etc
        self.k = k
        if method == 'classification':
            self.method = self.classification
        elif method == 'regression':
            self.method = self.regression
        else:
            print('ERROR: pick from regression or classification')
        self.method_label = method

    def fit(self, features, labels):
        # for classification
        # predict argmax (sum (w_i * 1[f(w_i) = y]))
        # w_i = 1/(d(x_q, x_i)^2)
        self.features = features
        self.labels = labels
        if len(labels.shape) > 1:
            self.num_out = labels.shape[1]
        else:
            self.num_out = 1

    def prediction(self, row):
        # find the k nearest data element
        k_closest = [None for _ in range(self.k)]
        for i, f in enumerate(self.features):
            d = distance(row, f)
            if k_closest[-1] is None or k_closest[-1][0] > d:
                # the loop is here to speed up computation (most of the above will be false, I think)
                for j in range(self.k):
                    if k_closest[j] is None or k_closest[j][0] > d:
                        # so then k_closest[0] will be the closest, up to k_closests[k] (the furthest)
                        # insert at position j, shifting all other elements up and popping the last (no longer in k closest)
                        k_closest.insert(j, [d, self.labels[i]])
                        k_closest.pop()
                        # don't fill up k_closest!
                        break
        # now, do the classification / regression
        return self.method(k_closest)

    @staticmethod
    def classification(k_closest):
        # could do weighting here
        classes = np.array([c for [_, c] in k_closest])
        # remove None values
        classes = classes[classes != np.array(None)]
        unique, counts = np.unique(classes, return_counts=True)
        return unique[np.argmax(counts)]

    @staticmethod
    def regression(k_closest):
        # could do more complex weighting - right now, just do a simple average
        # (eg inverse distance weighted average)
        # if there are many None's, k_closest may not be filled up
        # print('k_closest:', k_closest)
        classes = []
        # remove None values
        for k in k_closest:
            if not k is None:
                classes.append(k[1])

        # print(classes, np.mean(classes))
        return np.mean(classes)

    def predict(self, data):
        pred = np.apply_along_axis(self.prediction, 1, data)

        if self.method_label == 'classification':
            if self.num_out > 1:
                # need to encode the output
                # this will only work for integer outputs
                encode_pred = np.zeros((data.shape[0], self.num_out))
                for i, p in enumerate(pred):
                    encode_pred[i, int(p)] = 1
                return encode_pred
            else:
                return pred
        else: # regression
            return pred

if __name__=='__main__':
    import preprocess
    import risk
    from cross_validation import cross_validation

    print('processing data...')
    usecols = [i for i in range(0,26)] + [i for i in range(87,98)] + [161, 163] + [i for i in range(219,228)] + [279]
    X, y = preprocess.process_data(usecols=usecols,
        collapse=True, normalize=True, encode=False,
        predict_missing=True, k_predict=3)

    print('performing cross-validation...')
    models = [kNN(i) for i in range(1,10)]
    r = cross_validation(X, y, models)

    print(r)
    k = np.argmin(r) + 1

    print('evaluated best model (k:',k,')...')
    partitioned_data = preprocess.partition_data(X, y, partitions=[0.2,0.8])

    train = partitioned_data[1]
    valid = partitioned_data[0]

    model = kNN(k)

    print(train[0].shape, train[1].shape)
    model.fit(train[0], train[1])
    p = model.predict(valid[0])

    from evaluate import evaluate
    print(valid[1].shape, p.shape)
    evaluate(valid[1], p)
