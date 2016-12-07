import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
      scores = W.dot(X[:, i])
      correct_class_score = scores[y[i]]
      for j in range(num_classes):
          if j == y[i]:
              continue
          margin = scores[j] - correct_class_score + 1
          if margin > 0:
              loss += margin
              # compute gradients
              dW[j, :] += X[:, i].T  # transport column to row
              dW[y[i], :] -= X[:, i].T
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # dW
  dW /= num_train
  dW += reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    D = X.shape[0]
    num_classes = W.shape[0]
    num_train = X.shape[1]
    scores = W.dot(X)
    correct_scores = scores[y, np.arange(num_train)]  # (1, N)
    margins = scores - correct_scores + 1
    margins[y, np.arange(num_train)] = 0
    margins[margins < 0] = 0
    loss = np.sum(margins)
    loss /= num_train
    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)
    # compute gradient
    margins[margins > 0] = 1.0
    # compute col sum for dw_i update
    col_sum = np.sum(margins, axis=0)
    margins[y, range(num_train)] = -col_sum[range(num_train)]
    dW = np.dot(margins, X.T)
    dW /= num_train
    dW += reg*W

    return loss, dW
