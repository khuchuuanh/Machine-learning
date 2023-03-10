from __future__ import  division, print_function, unicode_literals
import math
import numpy as np
import matplotlib.pyplot as plt

def grad1(x):
    return 2*x +5*np.cos(x)

def cost1(x):
    return x**2 + 5*np.sin(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new  = x[-1] -eta*grad1(x[-1])
        if abs(grad1(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

#(x1, it1) = myGD1(.1, -5)
#(x2, it2) = myGD1(.1, 5)
#print('Solution x1 = %f, cost = %f, obtained after %d iteration'%(x1[-1], cost(x1[-1]), it1))
#print('Solution x1 = %f, cost = %f, obtained after %d iteration'%(x2[-1], cost(x1[-1]), it2))

# Optimize loss function of gradient descent

X = np.random.randn(1000,1)
y  = 4 + 3*X + .2* np.random.randn(1000, 1)

one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula : w = ', w_lr.T)

w = w_lr
print(w)
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0,1,2, endpoint= True)
y0 = w_0 +w_1*x0

plt.plot(X.T, y.T, 'b.')
plt.plot(x0, y0, 'y',linewidth = 2)
plt.axis([0,1,0,10])
#plt.show()

def grad(w):
    N = Xbar.shape[0]
    return 1/N *Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2

def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p) -cost(w_n)) / (2*eps)
    return g

def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))


def GD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)

w_init = np.array([[2], [1]])
(w1, it1) = GD(w_init, grad, 1)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))

# check convergence

def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new))/len(theta_new) <1e-3

def GD_momentum(theta_init, grad, eta, gramma):
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gramma*v_old  + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        if has_converged(theta_new, grad):
            break

        theta.append(theta_new)
        v_old = v_new
    return theta

def sgrad(w, i , rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

def SGD(w_init, grad, eta):
    w = [w_init]
    w_last_check  = w_init
    iter_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count% iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check) / len(w_init) < 1e-3:
                    return w
                w_last_check  = w_this_check
    return w
