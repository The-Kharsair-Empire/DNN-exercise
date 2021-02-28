import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *
from fundamental_neural_networks.logistic_regression.logistic_regression_shallow_NN import check_loaded_data, check_dimension, preprocess_image, load_dataset
from fundamental_neural_networks.deep_neural_network_modules.deep_neural_network_functional_module_implmentation import *
#from numba import jit, cuda, vectorize, njit

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def two_layer_model(X, Y, layers_dims = (12288, 7, 1), learning_rate = 0.0075, num_iterations = 3000, print_cost = False, plt_enable = False):
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameter_2layer_nn(n_x, n_h, n_y)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation='sigmoid')

        cost = compute_cost(A2, Y)

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    if plt_enable:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters


def dnn_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations = 3000, print_cost = False, plt_enable = False):
    np.random.seed(1)
    costs = []

    parameters = initialize_parameter_nlayer_nn(layer_dims)

    for i in range(num_iterations):
        AL, caches = linear_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = linear_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    if plt_enable:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters

def test_with_your_own_image(num_px, parameters, classes, image_name = "my_image1.jpg"):

    my_image =  image_name# change this to the name of your image file
    my_label_y = [0]  # the true class of your image (1 -> cat, 0 -> non-cat)
    ## END CODE HERE ##

    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False, mode='RGB'))
    my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px * num_px * 3, 1))
    my_predicted_image = predict(my_image, my_label_y, parameters)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")

def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
    check_loaded_data(train_x_orig, train_y, classes, True)
    check_dimension(train_x_orig, train_y, test_x_orig, test_y, classes)
    train_x, test_x = preprocess_image(train_x_orig, test_x_orig, 255, True)

    parameters = two_layer_model(train_x, train_y, layers_dims=(12288, 7, 1), num_iterations=2500, print_cost=True, plt_enable=True)
    predictions_train = predict(train_x, train_y, parameters)
    predictions_test = predict(test_x, test_y, parameters)

    parameters = dnn_model(train_x, train_y, layer_dims=[12288, 20, 7, 5, 1], num_iterations=2500, print_cost=True, plt_enable=True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)

    print_mislabeled_images(classes, test_x, test_y, pred_test)

if __name__ == '__main__':
    main()