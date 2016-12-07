############################################################################
#
# Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:

Authors: zhouxing(@ict.ac.cn)
Date:    2016/12/06 09:40:59
File:    svm.py
"""
import numpy as np
from cs231n.data_utils import gen_train_val_test
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized
import time
from cs231n.classifiers import LinearSVM
def evaluation(clf, x, y):
    y_pred = clf.predict(x)
    acc_train = np.mean(y == y_pred)
    return acc_train
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = gen_train_val_test(49000, 1000, 1000)
    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(10, 3073) * 0.01
    start = time.clock()
    loss, grad = svm_loss_naive(W, X_train, y_train, 0.00001)
    end = time.clock()
    print "svm_loss_naive: %f s" % (end - start)
    print 'loss: %f' % (loss, )
    start = time.clock()
    loss1, grad = svm_loss_vectorized(W, X_train, y_train, 0.00001)
    end = time.clock()
    print "svm_loss_vectorized: %f s" % (end - start)
    print 'loss: %f' % (loss1, )
    svm = LinearSVM()
    tic = time.time()
    loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=3e4,
                                  num_iters=2000,batch_size=200, verbose=True)
    acc_train = evaluation(svm, X_train, y_train)
    acc_val = evaluation(svm, X_val, y_val)
    acc_test = evaluation(svm, X_test, y_test)
    print 'Train acc :{} Validation :{} Test :{}'.format(acc_train, acc_val, acc_test)
    toc = time.time()
    print 'That took %fs' % (toc - tic)


if __name__ == '__main__':
    main()
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
