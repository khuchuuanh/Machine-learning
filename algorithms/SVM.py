from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cvxopt import  matrix, solvers
from sklearn.svm import SVC

means = [[2,2],[4,2]]
cov = [[.3,.2],[.2,.3]]
#print(means)
#print(cov)

N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov,N)
X  = np.concatenate((X0.T, X1.T), axis =1)
print('X0', X0)
print(X1)


y = np.concatenate((np.ones((1,N)), -1*np.ones((1,N))), axis = 1)

V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V))
p = matrix(-np.ones((2*N, 1)))
G = matrix(-np.eye(2*N))
h = matrix(np.zeros((2*N, 1)))
A = matrix(y)
b = matrix(np.zeros((1,1)))
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)
l = np.array(sol['x'])

#print('lambda = ')
#print(l.T)

epsilon = 1e-6
S = np.where(1 >epsilon)[0]
VS = V[:,S]
XS = X[:,S]
yS = y[:,S]
lS = l[S]

print(VS)

w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)

y1 = y.reshape((2*N,))
X1 = X.T
clf = SVC(kernel = 'linear', C = 1e5)
clf.fit(X1, y1)
w = clf.coef_
b = clf.intercept_

print('w = ' , w)
print('b = ',b )
