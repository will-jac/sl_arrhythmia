import numpy as np

def sigmoid(z):
    # print('sigmoid:', z)
    # print('after:', 1 / (1 + np.exp(-z)))
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z, delta):
    # print('sigmoid grad:', z)
    # z -> output of sigmoid initially
    # print('after:', s * (1 - s))
    return delta * z * (1 - z)

def ReLU(z):
    # print('ReLU:', z)
    return np.maximum(z, 0)

def ReLU_grad(z, delta):
    delta[z == 0] = 0
    return delta

def softmax(z):
    # print('softmax:', z)
    tmp = z - z.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=z)
    z /= z.sum(axis=1)[:, np.newaxis]
    return z
    # return np.exp(z) / np.sum(np.exp(z), axis=0)

# https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
def softmax_grad(z, delta):
    print('softmax grad:', z)
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = z.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def log_loss(predict, true):
    print('log loss:', predict, true)
    #(-1/m) * sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))
    return -1 * np.mean(true * np.log(predict) + (1-true) * np.log(1-predict))

def mse(predict, true, axis=0):
    # print('mse:') #, predict, true)
    # print('after:', np.sum(np.square(predict - true), axis=axis) / 2)
    return ((true - predict) ** 2).mean() / 2
# def mse_grad(predict, true, predict_grad):
#     print('mse grad:', predict, true)
#     # TODO: check that I vectorized this correctly (I don't think I did)
#     p = predict_grad(predict)
#     # print(p.shape, p)
#     print('after:', np.sum((predict - true) * p, axis=0))
#     # print(predict)
#     # print(true)
#     return np.sum((predict - true) * p, axis=0)

def mse_grad(predict, true, unused):
    # from ML class notes
    return np.sum((predict - true), axis=0)

#region old
# def forward_pass(X, Y, weights, bias, n_hidden, hidden_activation, output_activation, loss):
#     # https://towardsdatascience.com/back-propagation-414ec0043d7
#     # http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf

#     A = [0]*(n_hidden+1)
#     Z = [0]*(n_hidden+1)

#     # feed into input layer
#     # setting this as the last element of A makes it accessed by A[i-1] => A[-1] in the
#     # first iteration of the below loop
#     # basically, this is priming the loop for the hidden layers
#     A[-1] = np.array(X)

#     # forward propogation through hidden layers
#     for i in range(0, n_hidden+1):
#         # matrix operation
#         # do a matrix mulitply (dot product) on value * weight
#         # add bias (a row vector) for the layer
#         # gives layer output, Z, and activations, A
#         Z[i] = np.matmul(A[i-1], weights[i]) + bias[i]
#         A[i] = hidden_activation(Z[i])
#         # print('layer', i, A[i-1].shape, self.weights[i].shape, self.bias[i], '->', A[i].shape)

#         # then, feed A into the next layer (loop)

#         # during the forward pass, compute the local gradient
#         # (for each training example)
#         # dW[i] = np.apply_along_axis(self.hidden_activation_grad, 0, A[i])
#         # print('dw',i,':',dW[i].shape)
#         # dB[i] = self.weights[i].T

#     # do the output layer ( no output bias term! )
#     Z[-1] = np.matmul(A[-2], weights[-1])
#     A[-1] = output_activation(Z[-1])

#     # One-hot encode the output
#     # Accounts for non-binary output variables
#     P = np.zeros(A[-1].shape)
#     P[:, np.argmax(A[-1], 1)] = 1

#     # compute empirical risk
#     # E = log_loss(P, Y)
#     # sum allows for non-binary case (I think)
#     # C = - 1/2 * np.sum(np.square(Y - P), axis=0)
#     C = loss(P, Y)

#     return C, Z, A, P

# def backward_pass(C, P, Z, A, weights, n_hidden, learning_rate, hidden_activation_grad, output_activation_grad, loss_grad):
#     dC = [0]*(n_hidden+1)
#     dW = [0]*(n_hidden+1)
#     dB = [0]*(n_hidden+1)

#     dC[-1] = loss_grad(P, Y, output_activation_grad)
#     # in order to update our weights, we need to take the mean
#     dW[-1] = np.mean(dC[-1] * A[-1], axis=0)
#     weights[-1] += learning_rate * dW[-1]
#     downstream = np.matmul(dC[-1], weights[-1].T)
#     for i in range(n_hidden-1, 0, -1):
#         # dC -> delta
#         # print('downstream:', downstream.shape, downstream)
#         dC[i] = hidden_activation_grad(Z[i]) * downstream
#         dW[i] = np.mean(dC[i] * A[i], axis=0)
#         weights[i] += learning_rate * dW[i]
#         downstream = np.matmul(dC[i], weights[i].T)

#         # dC[i] = self.loss_grad(A[i], Y, self.hidden_activation_grad)
#         # dW[i] = np.mean(dC[i] * A[i-1], axis=0)
#         # print('dW', i, dW[i].shape)
#         # print(dW[i])
#         # # TODO: add learning rate
#         # self.weights[i] += dW[i]

#         # error[i] = np.matmul(self.weights[i+1].T * error[i+1], dW[i])
#         # print('error', i, ':')
#         # print(error[i])
#         # print('weights', i, ':')
#         # print(self.weights[i])
#         # self.weights[i] += error[i]
#         # print('new weights', i, ':')
#         # print(self.weights[i])
#     return weights
#endregion
class FFNN():

    # layers: [input_size, hidden_1_size, ..., output_size]
    # note: each layer internally will have a bias applied
    def __init__(self, hidden_layers,
            learning_rate = 0.05, num_iterations = 200,
            batch_size = 32,
            hidden_activation='sigmoid', output_activation='softmax',
            alpha = 0.001):

        if type(hidden_layers) is int:
            hidden_layers = [hidden_layers]
        else:
            hidden_layers = list(hidden_layers)

        if len(hidden_layers) < 1:
            print('error: must provide at least one hidden layer!')
            return

        self.n_hidden = len(hidden_layers)

        self.hidden_layers = hidden_layers

        # we will construct the weights and bias when we know the size of the input and outputs

        self.hidden_activation = sigmoid
        self.hidden_activation_grad = sigmoid_grad
        self.output_activation = softmax
        self.output_activation_grad = softmax_grad

        self.loss_fun = mse
        self.loss_grad = mse_grad

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        self.alpha = alpha

    def _initialize(self, y, layer_sizes):

        self.n_outputs = y.shape[1]
        self.n_layers = len(layer_sizes)

        self.weights = [None] * (self.n_layers - 1)
        self.bias = [None] * (self.n_layers - 1)

        # Values are random in the range [-0.5, 0.5]

        for i in range(self.n_layers - 1):
            self.weights[i] = (np.random.random((layer_sizes[i], layer_sizes[i+1])) - 0.5)
            self.bias[i] = (np.random.random((1, layer_sizes[i+1])) - 0.5)

        print('weights shape:', [w.shape for w in self.weights])
        print('bias shape:   ', [b.shape for b in self.bias])

    def _forward_pass(self, A):
        # A already contains the input data as A[0]
        # print('A', 0, A[0].shape)
        # hidden layers
        for i in range(self.n_layers - 2):
            A[i+1] = np.matmul(A[i], self.weights[i]) + self.bias[i]
            A[i+1] = self.hidden_activation(A[i+1])
            # print('A', i+1, A[i+1].shape)

        A[-1] = np.matmul(A[-2], self.weights[-1]) + self.bias[-1]
        A[-1] = self.output_activation(A[-1])
        # print('A', i+1, A[i+1].shape)
        # print('A', -1, A[-1].shape)

        # print('A shapes:', [a.shape for a in A])
        return A

    def _compute_grad(self, l, m, A, delta, weight_grads, bias_grads):
        # print('computing grad for layer', l)
        weight_grads[l] = (np.matmul(A[l].T, delta[l]) + (self.alpha * self.weights[l])) / m
        bias_grads[l] = np.mean(delta[l], axis=0)

    def _backprop(self, loss, m, y, A, delta, weight_grads, bias_grads):
        # do the back propogation

        # first, we're going to regularize the loss with respect to the L2 norm of the weights
        # this is so high # of weights doesn't negatively impact the loss

        # first, flatten out the weights and dot them with themselves
        # this gives us sumweights^2 per layer
        values = 0
        for weight_mat in self.weights:
            flat_weight = weight_mat.ravel()
            values += np.dot(flat_weight, flat_weight)
        # L2 regularization added to loss
        loss += (0.5 * self.alpha) * values / m

        # actually do the backpropogation
        l = self.n_layers - 2
        delta[l] = A[-1] - y
        # gradient for the last layer
        self._compute_grad(l, m, A, delta, weight_grads, bias_grads)

        # move backward through the hidden layers
        for i in range(self.n_layers - 2, 0, -1):
            delta[i-1] = np.matmul(delta[i], self.weights[i].T)
            delta[i-1] = self.hidden_activation_grad(A[i], delta[i-1])

            self._compute_grad(i - 1, m, A, delta, weight_grads, bias_grads)

        return weight_grads, bias_grads

    def _fit(self, X, y, A, delta, weight_grads, bias_grads, layer_sizes):
        m = X.shape[0]

        n_batches = (m // self.batch_size) + 1
        if m % self.batch_size == 0:
            n_batches -= 1

        for i in range(self.num_iterations):
            start = 0
            accumulated_loss = 0.0
            for j in range(n_batches):
                # generate the batches
                end = min(start + self.batch_size, m)
                X_batch = X[start:end, :]
                y_batch = y[start:end, :]
                batch_size = end - start
                start = end

                # time for the ML
                # print(X_batch.shape, y_batch.shape)
                # do backpropogation
                A[0] = X_batch

                A = self._forward_pass(A)

                # compute the loss
                # print('backprop:', y.shape, A[-1].shape)
                loss = self.loss_fun(y_batch, A[-1])

                weight_grads, bias_grads = self._backprop(loss, batch_size, y_batch, A, delta, weight_grads, bias_grads)
                accumulated_loss += loss * batch_size

                # update the weights

                #TODO: use an optimizer, eg momentum
                # print('bias before update:')
                # print(self.bias)

                for w, w_g in zip(self.weights, weight_grads):
                    w -= self.learning_rate * w_g
                for b, b_g in zip(self.bias, bias_grads):
                    b -= self.learning_rate *  b_g

                # print('bias after update:')
                # print(self.bias)

            self.loss = accumulated_loss / X.shape[0]
            # self.cost_curve.append(self.cost)
            if i % 100 == 0:
                print('Iter:', i, 'cost:', self.loss)

            # TODO: stopping conditions

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        (m, n_features) = X.shape
        (_, n_features_out) = y.shape

        # print('n_features_out:', n_features_out)

        # TODO: check it it's already fitted / initalized or not
        layer_sizes = ([n_features] + self.hidden_layers + [n_features_out])
        self._initialize(y, layer_sizes)

        # TODO: initialize weight / bias gradiant arrays

        A = [X] + [None] * (len(layer_sizes) - 1)
        delta = [None] * (len(layer_sizes) - 1)

        weight_grad = [np.empty((n_in, n_out), dtype=float) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        bias_grad = [np.empty((n_out), dtype=float) for n_out in layer_sizes[1:]]

        self._fit(X, y, A, delta, weight_grad, bias_grad, layer_sizes)
        # self.optimize_batch(X, y, A, delta, weight_grad, bias_grad, layer_sizes)

        # self.propogate(X, Y)
        # print('final weights:')
        # print(self.weights)
        # print(self.bias)

    def predict(self, X):
        A = [np.array(X)] + [None] * (self.n_layers - 1)
        # output is the last layer here
        return self._forward_pass(A)[-1]

def auto_encoder_test():
    nn = FFNN((3), batch_size=2, learning_rate=0.5)
    X = [
        [0,0],
        [1,0],
        [0,1]
    ]
    y = [
        [0,0],
        [1,0],
        [0,1]
    ]
    test=[[1,1]]
    test_out = [[1,1]]
    nn.fit(X, y)
    print('prediction: ', nn.predict(test))

    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier((3))
    model.fit(X,y)
    print('prediction: ', model.predict(test))

def MLP_test():
    from preprocess import process_data, partition_data
    print('processing data...')
    X, y = process_data(collapse=False, encode=True, normalize=True, predict_missing=True, k_predict=3)
    [test, validate, train] = partition_data(X, y)

    print('fitting model... ')
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(1000, 2000, 1000, 100, 50), verbose=True)
    model.fit(train[0], train[1])

    valid_prob = model.predict_proba(validate[0])

    from cross_entropy import cross_entropy
    print(valid_prob.shape, validate[1].shape)
    print('cross entropy:', cross_entropy(validate[1], valid_prob))

    from risk import empirical_risk
    print('mse:', empirical_risk('mse', valid_prob, validate[1]))

if __name__ == "__main__":
    # MLP_test()

    from preprocess import process_data, partition_data
    print('processing data...')
    X, y = process_data(collapse=False, encode=True,
        normalize=True, predict_missing=True, k_predict=3)
    [test, validate, train] = partition_data(X, y)

    print('fitting model...')
    model = FFNN((1000, 2000, 1000, 100, 50), num_iterations=500)
    model.fit(train[0], train[1])

    valid_prob = model.predict(validate[0])

    # print(valid_prob.shape, validate[1].shape)
    # print(valid_prob[0:10])
    # print(validate[1][0:10])

    # print(valid_prob[20:30])
    # print(validate[1][20:30])

    # print(valid_prob[40:50])
    # print(validate[1][40:50])

    from risk import empirical_risk
    print('mse:', mse(valid_prob, validate[1]))
    # sample of training output and training


    from cross_entropy import cross_entropy
    print(valid_prob.shape, validate[1].shape)
    print('cross entropy:', cross_entropy(validate[1], valid_prob))

    # MLP_test()

    # from kFold import cross_validation
    # print('performing cross-validation...')
    # r = kFold.cross_validation(data, models)