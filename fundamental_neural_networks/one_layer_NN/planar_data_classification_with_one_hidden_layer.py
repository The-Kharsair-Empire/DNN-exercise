import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

def setup(plt_enable=False, print_enable=False):
    np.random.seed(1)
    X, Y = load_planar_dataset()
    if plt_enable:
        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
        print("plotting is enable, please close the image to continue executing the program")
        plt.show()
        print()

    if print_enable:
        shape_X = X.shape
        shape_Y = Y.shape

        sample_size = shape_X[1]

        print('The shape of X is: ' + str(shape_X))
        print('The shape of Y is: ' + str(shape_Y))
        print('I have m = %d training examples!' % (sample_size))

    return X, Y

#######################################
# sklearn build in logistic regression for contrast
#######################################
def simple_logistic_regression(X, Y, plt_enable=False, print_enable=False):
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")

    if plt_enable:
        plt.show()

    LR_predictions = clf.predict(X.T)

    if print_enable:
        print()
        print("Simple Logistic Regression using skleran built in lib")
        print('Accuracy of logistic regression: %d ' % float(
            (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
              '% ' + "(percentage of correctly labelled datapoints)")


def test_cases(print_enable=False):
    X_assess, Y_assess = layer_sizes_test_case()
    (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
    if print_enable:
        print()
        print("The size of the input layer is: n_x = " + str(n_x))
        print("The size of the hidden layer is: n_h = " + str(n_h))
        print("The size of the output layer is: n_y = " + str(n_y))

    n_x, n_h, n_y = initialize_parameters_test_case()
    parameters = initialize_parameters(n_x, n_h, n_y)
    if print_enable:
        print()
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    X_assess, parameters = forward_propagation_test_case()
    A2, cache = forward_propagate(X_assess, parameters)
    if print_enable:
        print()
        print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

    A2, Y_assess, parameters = compute_cost_test_case()
    if print_enable:
        print()
        print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

    parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
    grads = backward_propagate(parameters, cache, X_assess, Y_assess)
    if print_enable:
        print()
        print("dW1 = " + str(grads["dW1"]))
        print("db1 = " + str(grads["db1"]))
        print("dW2 = " + str(grads["dW2"]))
        print("db2 = " + str(grads["db2"]))

    parameters, grads = update_parameters_test_case()
    parameters = gradient_descent(parameters, grads)
    if print_enable:
        print()
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    X_assess, Y_assess = nn_model_test_case()
    parameters = nn_model(X_assess, Y_assess, 4, 1.2,  num_iter=10000, print_cost=print_enable)
    if print_enable:
        print()
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

    parameters, X_assess = predict_test_case()
    predictions = predict(parameters, X_assess)
    if print_enable:
        print()
        print("predictions mean = " + str(np.mean(predictions)))

#######################################
# Neural Network Model
#######################################
def layer_sizes(X, Y):
    n_x = X.shape[0] # input layer size
    n_h = 4 # hidden layer size
    n_y = Y.shape[0] # output layer size
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return params

def forward_propagate(X, params):

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, params):
    sample_size = Y.shape[1]
    cost = -1/sample_size * np.sum(np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y))
    cost = float(np.squeeze(cost))
    assert (isinstance(cost, float))

    return cost

def backward_propagate(params, cache, X, Y):
    sample_size = X.shape[1]
    W1 = params["W1"]
    W2 = params["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2-Y
    dW2 = 1/sample_size*np.dot(dZ2, A1.T)
    db2 = 1/sample_size*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1/sample_size*np.dot(dZ1, X.T)
    db1 = 1/sample_size*np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def gradient_descent(params, grads, alpha=1.2):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2

    new_params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return new_params


def nn_model(X, Y, n_h, learning_rate = 1.2, num_iter = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    params = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iter):
        A2, cache = forward_propagate(X, params)
        cost = compute_cost(A2, Y, params)

        grads = backward_propagate(params, cache, X, Y)
        params = gradient_descent(params, grads, learning_rate)

        if print_cost and i%1000 == 0:
            print ("Cost after iteration {}: {}".format(i, cost))

    return params


def predict(params, X):
    A2, cache = forward_propagate(X, params)

    return (A2 > 0.5)

#######################################
# Tuning hidden layer size
#######################################
def tuning_n_h(X, Y, ):
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iter=5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    plt.show()

#######################################
# Additional dataset can be added to the image folder and predict on those image using your learned w and b
#######################################
def additional_dataset(): # replace the X, Y in the main function with the return values from this function
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}
    ### (choose your dataset)
    dataset = "gaussian_quantiles"

    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    # make blobs binary
    if dataset == "blobs":
        Y = Y % 2

    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    return X, Y

def all_datasets():
    # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    i = 0
    plt.figure(figsize=(8, 16))
    for dataset in datasets:
        plt.subplot(4, 2, i + 1)
        i += 1
        plt.title(dataset)

        X, Y = datasets[dataset]
        X, Y = X.T, Y.reshape(1, Y.shape[0])


        if dataset == "blobs":
            Y = Y % 2

        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
        # Build a model with a n_h-dimensional hidden layer
        parameters = nn_model(X, Y, n_h=4, learning_rate=1.2, num_iter=10000, print_cost=False)
        plt.subplot(4, 2, i + 1)
        i += 1
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        plt.title(dataset + 'Classifier')
    plt.show()

def main():
    X, Y = setup(False, False)
   # X, Y = additional_dataset()
    simple_logistic_regression(X, Y)
    test_cases()

    parameters = nn_model(X, Y, n_h=4, learning_rate=1.2, num_iter=10000, print_cost=False)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()

    predictions = predict(parameters, X)
    print('Accuracy: %d' % float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    tuning_n_h(X, Y)

    all_datasets()




if __name__ == '__main__':
    main()

