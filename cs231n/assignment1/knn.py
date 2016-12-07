############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/12/05 21:28:49
File:    knn.py
"""
import sys
import numpy as np
import random
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import KNearestNeighbor
import datetime
def gen_train_test(num_training, num_test):
    """ generate training and testing dataset

    """
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    print 'Training data shape: ', X_train.shape
    print 'Training labels shape: ', y_train.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape

    # Subsample the data for more efficient code execution
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print X_train.shape, X_test.shape
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = gen_train_test(5000, 500)
    num_test = y_test.shape[0]
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    starttime = datetime.datetime.now()
    dists = classifier.compute_distances_one_loop(X_test)
    endtime = datetime.datetime.now()
    print (endtime-starttime).seconds
    print dists.shape
    y_test_pred = classifier.predict_labels(dists, k=5)
    # Compute and print the fraction of correctly predicted examples
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
if __name__ == '__main__':
    main()
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
