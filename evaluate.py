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

    try:
        print('mse:', mse(true, predict_prob).mean())
    except:
        print('mse failed')
    try:
        # print(valid_prob.shape, validate[1].shape)
        print('cross entropy:', cross_entropy(true, predict_prob)) #, is_binary))
    except:
        print('cross entropy failed')
    
    if is_binary:
        # y is alread encoded
        correct = true
        predict = predict_prob
    else:
        # one-hot encode valudate and probabilities
        correct = np.argmax(true, axis=1)
        predict = np.argmax(predict_prob, axis=1)
    print(correct.shape, predict.shape)

    try:
        print('accuracy:', accuracy_score(correct, predict))
    except:
        print('accuracy failed')
    try:
        print('confusion matrix:')
        print(confusion_matrix(correct, predict))
    except:
        print('confusion matrix failed')
