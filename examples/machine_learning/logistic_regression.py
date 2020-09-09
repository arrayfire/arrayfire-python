#!/usr/bin/env python

#######################################################
# Copyright (c) 2019, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause
########################################################

from mnist_common import display_results, setup_mnist

import sys
import time

import arrayfire as af
from arrayfire.algorithm import max, imax, count, sum
from arrayfire.arith import abs, sigmoid, log
from arrayfire.array import transpose
from arrayfire.blas import matmul, matmulTN
from arrayfire.data import constant, join, lookup, moddims
from arrayfire.device import set_device, sync, eval


def accuracy(predicted, target):
    _, tlabels = af.imax(target, 1)
    _, plabels = af.imax(predicted, 1)
    return 100 * af.count(plabels == tlabels) / tlabels.elements()


def abserr(predicted, target):
    return 100 * af.sum(af.abs(predicted - target)) / predicted.elements()


# Predict (probability) based on given parameters
def predict_prob(X, Weights):
    Z = af.matmul(X, Weights)
    return af.sigmoid(Z)


# Predict (log probability) based on given parameters
def predict_log_prob(X, Weights):
    return af.log(predict_prob(X, Weights))


# Give most likely class based on given parameters
def predict_class(X, Weights):
    probs = predict_prob(X, Weights)
    _, classes = af.imax(probs, 1)
    return classes


def cost(Weights, X, Y, lambda_param=1.0):
    # Number of samples
    m = Y.dims()[0]

    dim0 = Weights.dims()[0]
    dim1 = Weights.dims()[1] if len(Weights.dims()) > 1 else None
    dim2 = Weights.dims()[2] if len(Weights.dims()) > 2 else None
    dim3 = Weights.dims()[3] if len(Weights.dims()) > 3 else None
    # Make the lambda corresponding to Weights(0) == 0
    lambdat = af.constant(lambda_param, dim0, dim1, dim2, dim3)

    # No regularization for bias weights
    lambdat[0, :] = 0

    # Get the prediction
    H = predict_prob(X, Weights)

    # Cost of misprediction
    Jerr = -1 * af.sum(Y * af.log(H) + (1 - Y) * af.log(1 - H), dim=0)

    # Regularization cost
    Jreg = 0.5 * af.sum(lambdat * Weights * Weights, dim=0)

    # Total cost
    J = (Jerr + Jreg) / m

    # Find the gradient of cost
    D = (H - Y)
    dJ = (af.matmulTN(X, D) + lambdat * Weights) / m

    return J, dJ


def train(X, Y, alpha=0.1, lambda_param=1.0, maxerr=0.01, maxiter=1000, verbose=False):
    # Initialize parameters to 0
    Weights = af.constant(0, X.dims()[1], Y.dims()[1])

    for i in range(maxiter):
        # Get the cost and gradient
        J, dJ = cost(Weights, X, Y, lambda_param)

        err = af.max(af.abs(J))
        if err < maxerr:
            print('Iteration {0:4d} Err: {1:4f}'.format(i + 1, err))
            print('Training converged')
            return Weights

        if verbose and ((i+1) % 10 == 0):
            print('Iteration {0:4d} Err: {1:4f}'.format(i + 1, err))

        # Update the parameters via gradient descent
        Weights = Weights - alpha * dJ

    if verbose:
        print('Training stopped after {0:d} iterations'.format(maxiter))

    return Weights


def benchmark_logistic_regression(train_feats, train_targets, test_feats):
    t0 = time.time()
    Weights = train(train_feats, train_targets, 0.1, 1.0, 0.01, 1000)
    af.eval(Weights)
    sync()
    t1 = time.time()
    dt = t1 - t0
    print('Training time: {0:4.4f} s'.format(dt))

    t0 = time.time()
    iters = 100
    for i in range(iters):
        test_outputs = predict_prob(test_feats, Weights)
        af.eval(test_outputs)
    sync()
    t1 = time.time()
    dt = t1 - t0
    print('Prediction time: {0:4.4f} s'.format(dt / iters))


# Demo of one vs all logistic regression
def logit_demo(console, perc):
    # Load mnist data
    frac = float(perc) / 100.0
    mnist_data = setup_mnist(frac, True)
    num_classes = mnist_data[0]
    num_train = mnist_data[1]
    num_test = mnist_data[2]
    train_images = mnist_data[3]
    test_images = mnist_data[4]
    train_targets = mnist_data[5]
    test_targets = mnist_data[6]

    # Reshape images into feature vectors
    feature_length = int(train_images.elements() / num_train);
    train_feats = af.transpose(af.moddims(train_images, feature_length, num_train))
    test_feats = af.transpose(af.moddims(test_images, feature_length, num_test))

    train_targets = af.transpose(train_targets)
    test_targets = af.transpose(test_targets)

    num_train = train_feats.dims()[0]
    num_test = test_feats.dims()[0]

    # Add a bias that is always 1
    train_bias = af.constant(1, num_train, 1)
    test_bias = af.constant(1, num_test, 1)
    train_feats = af.join(1, train_bias, train_feats)
    test_feats = af.join(1, test_bias, test_feats)

    # Train logistic regression parameters
    Weights = train(train_feats, train_targets,
                    0.1,  # learning rate
                    1.0,  # regularization constant
                    0.01, # max error
                    1000, # max iters
                    True  # verbose mode
    )
    af.eval(Weights)
    af.sync()

    # Predict the results
    train_outputs = predict_prob(train_feats, Weights)
    test_outputs = predict_prob(test_feats, Weights)

    print('Accuracy on training data: {0:2.2f}'.format(accuracy(train_outputs, train_targets)))
    print('Accuracy on testing data: {0:2.2f}'.format(accuracy(test_outputs, test_targets)))
    print('Maximum error on testing data: {0:2.2f}'.format(abserr(test_outputs, test_targets)))

    benchmark_logistic_regression(train_feats, train_targets, test_feats)

    if not console:
        test_outputs = af.transpose(test_outputs)
        # Get 20 random test images
        display_results(test_images, test_outputs, af.transpose(test_targets), 20, True)

def main():
    argc = len(sys.argv)

    device  = int(sys.argv[1])      if argc > 1 else 0
    console = sys.argv[2][0] == '-' if argc > 2 else False
    perc    = int(sys.argv[3])      if argc > 3 else 60

    try:
        af.set_device(device)
        af.info()
        logit_demo(console, perc)
    except Exception as e:
        print('Error: ', str(e))


if __name__ == '__main__':
    main()
