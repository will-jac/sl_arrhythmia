import numpy as np

# Ideas for development / different options:
#
# Type of NN:
# - Recurrent
# - Feedforward
#
# Type of activation
# - ReLU
#
# Loss function
# - MSE
#
# Type of backpropogation
# - ?
#
# Meta-parameters
# - Input / output layer sizes
#

# basic code

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros(shape=(1, dim))
    b = 0

    assert(w.shape == (1, dim))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

def propogate(w, b, X, Y):
    m = X.shape[0]
    # print(m)
    print(X.shape, w.shape)
    # forward prop
    A = sigmoid(np.dot(X, w.T) + b)
    # print(A)
    # print(Y.shape)
    # print(A.shape)
    # print((Y * np.log(A)).shape)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))

    # backward prop
    dw = (1/m) * np.dot((A-Y).T, X)
    db = (1/m) * np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return dw, db, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):

    costs = []

    for i in range(num_iterations):
        # cost and gradient calculation
        dw, db, cost = propogate(w, b, X, Y)

        # update rule
        w -= learning_rate * dw
        b -= learning_rate * db

        # record costs
        if i % 100 == 0:
            costs.append(cost)

            print('Cost after iteration %i: %f' % (i,cost))

    return w, b, dw, db, costs

def predict(w, b, X):
    m = X.shape[0]
    Y_prediction = np.zeros((m,1))
    w = w.reshape(1, X.shape[1])

    # probability vector
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[0]):
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0

    assert(Y_prediction.shape == (m, 1))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
    print(X_train.shape)
    w, b = initialize_with_zeros(X_train.shape[1])

    # gradient descent
    w, b, dw, db, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print('train: {} %'.format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print('train: {} %'.format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

if __name__ == "__main__":
    import preprocess
    data = preprocess.process_data()
    [test, train] = preprocess.partition_data(data, [0.2,0.8])

    model(train[:,0:-1],train[:,-1],test[:,0:-1],test[:,-1], 2000, 0.005)


