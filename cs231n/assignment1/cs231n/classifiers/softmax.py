import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (C, D) containing weights.
  - X: A numpy array of shape (D, N) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0]
  num_train = X.shape[1]
  for i in range(num_train):
      f_i = W.dot(X[:, i])
      f_i -= np.max(f_i)  # normalized by substract maxmum value
      sum_i = np.sum(np.exp(f_i))
      loss += -f_i[y[i]] + np.log(sum_i)
      # compute gradient
      for j in range(num_classes):
          p = np.exp(f_i[j])/sum_i
          dW[j, :] += (p-(j==y[i]))*X[:, i]
  # average
  loss /= num_train
  dW /= num_train
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0]
  num_train = X.shape[1]
  f = np.dot(W, X)
  f -= np.max(f) # minus global maxmum value
  f_correct = f[y, range(num_train)]
  loss = -np.mean( np.log(np.exp(f_correct)/np.sum(np.exp(f), axis=0)) )
  # gradient
  p = np.exp(f)/np.sum(np.exp(f), axis=0)
  ind = np.zeros(p.shape)
  ind[y, range(num_train)] = 1
  dW = np.dot(p-ind, X.T)
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  return loss, dW

