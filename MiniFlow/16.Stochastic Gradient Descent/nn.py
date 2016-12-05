"""
Have fun with the number of epochs!

Be warned that if you increase them too much,
the VM will time out :)
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from miniflow import *

X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

# load data
data = load_boston()
X_ = data['data']
y_ = data['target']
# normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# neural network
l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 1000
m = X_.shape[0] # total number of examples

graph = topological_sort(feed_dict)
trainables = [W1, b2, W2, b2]

# step 4
for i in range(epochs):
    # step 1
    X_, y_ = shuffle(X_, y_)
    # reset value of Input
    X.value = X_
    y.value = y_

    # step 2
    forward_and_backward(graph)

    # step 3
    sgd_update(trainables)

    print('Epoch: ' + str(i) + ', Loss: ' + str(graph[-1].value))
