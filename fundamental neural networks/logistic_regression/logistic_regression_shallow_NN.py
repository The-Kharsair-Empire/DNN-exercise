import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import random


#######################################
# Some work to load and preprocess the image data into training and testing samples.
#######################################

def load_dataset():
    with h5py.File('datasets/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('datasets/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def check_loaded_trainset_data(image_data_set_orig, image_data_set_y, classes, plt_enable = False):
    print()
    print(check_loaded_trainset_data.__name__)
    index = random.randint(0, len(image_data_set_orig))
    plt.imshow(image_data_set_orig[index])
    print("randomly selected image at index: {}".format(index))
    print("shape of the image data: {}".format(image_data_set_orig.shape))
    print("y = " + str(image_data_set_y[:, index]) + ", it's a '" +
          classes[np.squeeze(image_data_set_y[:, index])].decode("utf-8") + "' picture.")
    if plt_enable:
        print("plotting is enable, please close the image to continue executing the program")
        plt.show()

def check_dimension(train_set_x_orig,  train_set_y, test_set_x_orig, test_set_y, classes):
    print()
    print(check_dimension.__name__)
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = test_set_x_orig[0].shape[0]

    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))

def preprocess_image(train_set_x_orig, test_set_x_orig, value_range=255., print_enable=False):
    #reshaping
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    #normalization
    train_set_x = train_set_x_flatten / value_range
    test_set_x = test_set_x_flatten / value_range
    if print_enable:
        print()
        print(preprocess_image.__name__)
        print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
        print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
        print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))
    return train_set_x, test_set_x

def setup():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    check_loaded_trainset_data(train_set_x_orig, train_set_y, classes)
    check_dimension(train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes)
    train_set_x, test_set_x = preprocess_image(train_set_x_orig, test_set_x_orig, 255., True)
    return train_set_x, train_set_y, test_set_x, test_set_y, classes, test_set_x_orig[0].shape[0], test_set_x_orig[0].shape[1]


#######################################
# Some Helper functions for logistic regression neural network
#######################################

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost_function(A, Y, sample_size):
    return -1/sample_size * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

def dJ_dw(X, A, Y, m):
    return 1/m * np.dot(X, (A - Y).T)

def dJ_db(A, Y, m):
    return 1/m * np.sum(A - Y)

def initialize_parameters(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


#######################################
# Logistic Regression Neural Network is defined below
#######################################

def forward_propagate(w, b, X, Y):
    sample_size = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = cost_function(A, Y, sample_size)

    dw = dJ_dw(X, A, Y, sample_size)
    db = dJ_db(A, Y, sample_size)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return {'dw': dw, 'db': db}, cost

def backward_propagate(w, b, X, Y, num_iter, alpha, print_enable=False):
    costs = []
    for i in range(num_iter):
        grads, cost = forward_propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']

        w -= alpha * dw
        b -= alpha * db

        if i % 100 == 0:
            costs.append(cost)

        if print_enable and i % 100 == 0:
            print ("Cost after iteration {}: {}".format(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X, print_enable=False):

    sample_size = X.shape[1]
    Y_pred = np.zeros((1, sample_size))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    if print_enable:
        print(X.shape)
        print(w.shape)
        print(A.shape)

    for i in range(A.shape[1]):
        Y_pred[0][i] = 1 if A[0][i] > 0.5 else 0

    assert (Y_pred.shape == (1, sample_size))

    return Y_pred

def logistic_regression_model(X_train, Y_train, X_test, Y_test, num_iter=2000, learning_rate=0.5, print_enable=False):
    w, b = initialize_parameters(X_train.shape[0])

    params, grads, costs = backward_propagate(w, b, X_train, Y_train, num_iter, learning_rate, print_enable)

    w = params['w']
    b = params['b']

    Y_pred_train = predict(w, b, X_train)
    Y_pred_test = predict(w, b, X_test)

    if print_enable:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_pred_test,
         "Y_prediction_train": Y_pred_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iter}

    return d

#######################################
# Assessment of the model training result
#######################################

def check_learning_result(test_set_x, test_set_y, d, classes, original_dim_x, original_dim_y, plt_enable=False, plt_learning_curve=False):
    print()
    print(check_learning_result.__name__)
    index = random.randint(0, test_set_y.shape[1])
    plt.imshow(test_set_x[:, index].reshape((original_dim_x, original_dim_y, 3)))
    print("randomly selected image at index: {}".format(index))
    print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[
        int(d['Y_prediction_test'][0, index])].decode("utf-8") + "\" picture.")

    if plt_enable:
        print("plotting is enable, please close the image to continue executing the program")
        plt.show()

    if plt_learning_curve:
        costs = np.squeeze(d['costs'])
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(d["learning_rate"]))
        print("plotting is enable, please close the plot to continue executing the program")
        plt.show()


#######################################
# Learning rate analysis
#######################################

def analyze_learning_rate(learning_rates, train_set_x, train_set_y, test_set_x, test_set_y):
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = logistic_regression_model(train_set_x, train_set_y, test_set_x, test_set_y, 1500, i, True)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


#######################################
# Additional images can be added to the image folder and predict on those image using your learned w and b
#######################################

def predict_on_other_image(img_name, w, b, plt_enable):

    fname = "images/" + img_name
    image = np.array(ndimage.imread(fname, flatten=False))
    image = image/255.
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(w, b, my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

    if plt_enable:
        plt.show()


if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y, classes, original_dim_x, original_dim_y = setup()
    model_result = logistic_regression_model(train_set_x, train_set_y, test_set_x, test_set_y, 2000, 0.005, True)
    check_learning_result(test_set_x, test_set_y, model_result, classes, original_dim_x, original_dim_y, True, True)

    analyze_learning_rate([0.01, 0.001, 0.0001], train_set_x, train_set_y, test_set_x, test_set_y)