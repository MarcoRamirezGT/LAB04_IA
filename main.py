# referencia: https: // github.com/INF800/Numpy-FFNN/blob/master/FFNN.py

import numpy as np
from sklearn.datasets import make_classification


def nonlin(x, deriv=False):
    if deriv == True:
        return (x*(1-x))
    return (1/(1+np.exp(-x)))


x, y = make_classification(n_samples=100, n_features=3,
                           n_informative=3, n_redundant=0, n_classes=2)

np.random.seed(1)

w1 = 2*np.random.random((3, 1)) - 1
w2 = 2*np.random.random((1, 100)) - 1

for j in range(60000):

    l0 = x
    l1 = nonlin(np.dot(l0, w1))
    l2 = nonlin(np.dot(l1, w2))

    # BACKPROPGATION
    l2_error = y - l2

    # printing status
    if(j % 10000) == 0:
        print('Error : ' + str(np.mean(np.abs(l2_error))))

    # calculte deltas
    l2_delta = l2_error*nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(w2.T)
    l1_delta = l1_error*nonlin(l1, deriv=True)

    # update our synapses
    w2 += l1.T.dot(l2_delta)
    w1 += l0.T.dot(l1_delta)

print('Output after training')
print(l2)
