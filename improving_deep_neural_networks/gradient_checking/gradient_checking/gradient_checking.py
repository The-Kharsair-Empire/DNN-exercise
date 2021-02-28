
import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

def forward_propagation(x, theta):
    return np.multiply( x,  theta)

def backward_propagation(x, theta):
    return x

def gradient_check(x, theta, epsilon = 1e-7):
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    J_plus = forward_propagation(x, theta_plus)
    J_minus = forward_propagation(x, theta_minus)

    grad_approx = (J_plus - J_minus) / (2 * epsilon)

    grad = backward_propagation(x, theta)

    diff = np.linalg.norm(grad - grad_approx) / (np.linalg.norm(grad) + np.linalg.norm(grad_approx))

    if diff < epsilon:
        print('correct gradient')
    else:
        print('wrong gradient')

    return diff

def fwd_prop_n(X, Y, params):
    m = X.shape[1]
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache

def bwd_prop_n(X, Y, cache):
        
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) * 2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 4./m * np.sum(dZ1, axis=1, keepdims = True) #db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    grads = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
   
    
    return grads

def grad_check_n(params, grads, X, Y, e = 1e-7):
    params_vals, _ = dictionary_to_vector(params)
    grad = gradients_to_vector(grads)
    num_params = params_vals.shape[0]
    J_plus = np.zeros((num_params, 1))
    J_minus = np.zeros((num_params, 1))
    grad_approx = np.zeros((num_params, 1))

    for i in range(num_params):
        theta_p = np.copy(params_vals)
        theta_p[i][0] += e
        J_plus[i], _ = fwd_prop_n(X, Y, vector_to_dictionary(theta_p))
        
        theta_m = np.copy(params_vals)
        theta_m[i][0] -= e
        J_minus[i], _ = fwd_prop_n(X, Y, vector_to_dictionary(theta_m))

        grad_approx[i] = (J_plus[i] - J_minus[i])/(2*e)

    diff = np.linalg.norm(grad - grad_approx) / (np.linalg.norm(grad) + np.linalg.norm(grad_approx))
    if diff > 2e-7:
        print ("There is a mistake in the backward propagation! difference = " + str(diff) )
    else:
        print ("Your backward propagation works perfectly fine! difference = " + str(diff) )
    return diff
    
def main():
    x, theta = 2, 4
    J = forward_propagation(x, theta)
    print ("J = " + str(J))

    x, theta = 2, 4
    dtheta = backward_propagation(x, theta)
    print ("dtheta = " + str(dtheta))

    x, theta = 2, 4
    difference = gradient_check(x, theta)
    print("difference = " + str(difference))

    
    X, Y, parameters = gradient_check_n_test_case()
    cost, cache = fwd_prop_n(X, Y, parameters)
    gradients = bwd_prop_n(X, Y, cache)
    difference = grad_check_n(parameters, gradients, X, Y)


if __name__ == '__main__':
    main()