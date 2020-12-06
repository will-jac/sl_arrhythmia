import numpy as np

def cross_entropy(y, p, y_is_binary=False, eps=1e-15):
    if y_is_binary:
        # y = list, change to [[y[0]],[y[1]], ... ] (eg column vector)
        y = np.reshape(np.array(y), newshape=(-1,1))

        # p is stored as [prob_0, prob_1],
        # so transform y to [1 - true_label, true_label]
        # if true_label = 0
        #  -> [1,0] -> correct, incorrect
        # if true_label = 1
        #  -> [0,1] -> incorrect, correct
        # from this, we can do y * log(p) to get
        # (true * log(predict) + (1 - true) * (1 - log(predict)))
        y = np.append(1 - y, y, axis=1)


    p = np.array(p)
    # prevent log (0) errors
    # sum will still == 1 becaue 1 - 1e-15, 0 + 1e-15 -> 1
    # (adding / subtracting by the same amount)
    p = np.clip(p, eps, 1-eps)

    # actually do the cross entropy
    # sum( y_i * ln(p_i) + (1-y_i) * ln(1-p_i) )
    log_loss = (y * np.log(p)).sum(axis=1)

    # -1/m * loss
    return -1 * np.average(log_loss)
