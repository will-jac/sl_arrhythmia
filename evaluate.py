from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

from cross_entropy import cross_entropy

def mse(predict, true, axis=0):
    return np.sum(np.square(predict - true), axis=axis) / 2

def evaluate(true, predict_prob):

    if len(predict_prob.shape) == 1:
        is_binary = True
    else:
        is_binary = False

    print('mse:', mse(true, predict_prob).mean())

    # print(valid_prob.shape, validate[1].shape)
    print('cross entropy:', cross_entropy(true, predict_prob, is_binary))

    # one-hot encode valudate and probabilities
    correct = np.argmax(true, axis=1)
    predict = np.argmax(predict_prob, axis=1)
    print(correct.shape, predict.shape)

    print('accuracy:', accuracy_score(correct, predict))

    print('confusion matrix:', confusion_matrix(correct, predict))