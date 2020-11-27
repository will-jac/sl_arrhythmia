import numpy as np

def mse(predict, true):
    return np.square(predict - true)

def zero_one(predict, true):

    def eq(v):
        if (v != 0):
            return 0
        return 1

    return np.mean(np.apply_along_axis(eq, 1, predict - true))

def empirical_risk(loss, predict, true):
    if loss=='mse':
        return np.mean(mse(predict, true))
    if loss=='0/1':
        return np.mean(zero_one(predict, true))
    else:
        print('error calcing risk: bad loss')
        return None