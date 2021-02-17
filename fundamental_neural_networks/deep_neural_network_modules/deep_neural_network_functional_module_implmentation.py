import numpy as np
import h5py
import matplotlib.pyplot as plt
from fundamental_neural_networks.deep_neural_network_modules.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

def initialize_parameter_2layer_nn(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def initialize_parameter_nlayer_nn(layer_dims): #layer_dims is a list of number of nodes in each layer
    np.random.seed(3)
    parameters = {}

    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    cache = (A, W, b)

    assert (Z.shape == (W.shape[0], A.shape[1]))
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)
    return A, cache

def linear_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) //2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T))

    cost = np.squeeze(cost)
    assert cost.shape == ()

    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def linear_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    m = AL.shape[1]

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        grads['dA' + str(l)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')

    return grads

def update_parameters(params, grads, learning_rate):
    L = len(params) // 2

    for l in range(L):
        params['W' + str(l + 1)] = params['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        params['b' + str(l + 1)] = params['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return params

def main():
    # parameters = initialize_parameter_2layer_nn(2, 2, 1)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    # parameters = initialize_parameter_nlayer_nn([5, 4, 3])
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    # A, W, b = linear_forward_test_case()
    # Z, linear_cache = linear_forward(A, W, b)
    # print("Z = " + str(Z))

    # A_prev, W, b = linear_activation_forward_test_case()
    # A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
    # print("With sigmoid: A = " + str(A))
    # A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
    # print("With ReLU: A = " + str(A))

    # X, parameters = L_model_forward_test_case_2hidden()
    # AL, caches = linear_model_forward(X, parameters)
    # print("AL = " + str(AL))
    # print("Length of caches list = " + str(len(caches)))

    # Y, AL = compute_cost_test_case()
    # print("cost = " + str(compute_cost(AL, Y)))

    # dZ, linear_cache = linear_backward_test_case()
    # dA_prev, dW, db = linear_backward(dZ, linear_cache)
    # print("dA_prev = " + str(dA_prev))
    # print("dW = " + str(dW))
    # print("db = " + str(db))

    # dAL, linear_activation_cache = linear_activation_backward_test_case()
    # dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="sigmoid")
    # print("sigmoid:")
    # print("dA_prev = " + str(dA_prev))
    # print("dW = " + str(dW))
    # print("db = " + str(db) + "\n")
    # dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="relu")
    # print("relu:")
    # print("dA_prev = " + str(dA_prev))
    # print("dW = " + str(dW))
    # print("db = " + str(db))

    # AL, Y_assess, caches = L_model_backward_test_case()
    # grads = linear_model_backward(AL, Y_assess, caches)
    # print_grads(grads)

    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads, 0.1)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

if __name__ == '__main__':
    from testCases_v4a import *
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    np.random.seed(1)

    main()