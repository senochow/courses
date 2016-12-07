############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/12/07 19:05:33
File:    softmax.py
"""
import sys
import numpy as np
from cs231n.data_utils import gen_train_val_test
from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.classifiers import Softmax
from svm import evaluation
import time


def run_softmax_naive(X_train, y_train):
    # Generate a random softmax weight matrix and use it to compute the loss.
    W = np.random.randn(10, 3073) * 0.0001
    start = time.clock()
    loss, grad = softmax_loss_vectorized(W, X_train, y_train, 0.0)
    end = time.clock()
    print "softmax_loss_naive: %f s" % (end - start)
    # As a rough sanity check, our loss should be something close to -log(0.1).
    print 'loss: %f' % loss
    print 'sanity check: %f' % (-np.log(0.1))

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = gen_train_val_test(49000, 1000, 1000)
    #run_softmax_naive(X_train, y_train)
    softmax = Softmax()
    tic = time.time()
    softmax.train(X_train, y_train, learning_rate=2.782559e-06, reg=1e3,num_iters=3000,batch_size=200, verbose=True)

    acc_train = evaluation(softmax, X_train, y_train)
    acc_val = evaluation(softmax, X_val, y_val)
    acc_test = evaluation(softmax, X_test, y_test)
    print 'Train acc :{} Validation :{} Test :{}'.format(acc_train, acc_val, acc_test)
    toc = time.time()
    print 'That took %fs' % (toc - tic)


if __name__ == '__main__':
    main()
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
