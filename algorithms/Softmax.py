import numpy as np
from scipy import sparse

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis= 0)

def softmax_stabel(Z):
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims= True))
    A = e_Z/ e_Z.sum(axis = 0)
    return A

N  = 2
d = 2
C = 3

X = np.random.randn(d, N)
y = np.random.randint(0,3, (N,))

def convert_labels(y, C = C):
    Y = sparse.coo_matrix((np.ones_like(y),
                           (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    
    return Y

Y  = convert_labels(y, C)

def cost(X, y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y*np.log(A))

def grad(X,Y, W):
    A = softmax((W.T.dot(X)))
    E = A - Y
    return X.dot(E.T)

def numerical_grad(X, Y, W, cost):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i,j] += eps
            W_n[i, j] -= eps
            g[i,j] = (cost(X, Y, W_p) - cost(X, Y, W_n)) / (2*eps)
    return g
W_init = np.random.randn(d, C)
g1 = grad(X, Y, W_init)
g2 = numerical_grad(X, Y, W_init, cost)
#print(np.linalg.norm(g1 - g2))

def softmax_regression(X, y, W_init, eta, tol = 1e-4, max_count = 10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]

    count = 0
    check_afer  =20
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai  =softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta * xi.dot((yi - ai).T)
            count += 1

            if count % check_afer == 0:
                if np.linalgnorm(W_new - W[-check_afer]) < tol:
                    return W
            W.append(W)

    return W
eta = .05
d = X.shape[0]
W_init =np.random.randn(d,C)

W = softmax_regression(X, y, W_init, eta)

def pred(W, X):
    A = softmax_stabel(W.T.dot(X))
    return np.argmax(A, axis= 0)

                     
