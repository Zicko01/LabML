""" ps4_implementation.py

PUT YOUR NAME HERE:
Defne Milen Güler


Complete the classes and functions
- svm_qp
- plot_svm_2d
- neural_network
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2019
"""
import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch

# This is already implemented for your convenience
class svm_sklearn():
    """ SVM via scikit-learn """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
        self.clf = sklearn.svm.SVC(C=C,
                                   kernel=kernel,
                                   gamma=1./(1./2. * kernelparameter ** 2),
                                   degree=int(kernelparameter),
                                   coef0=kernelparameter)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.X_sv = X[self.clf.support_, :]
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)

def sqdistmat(X, Y=False):
    if Y is False:
        X2 = sum(X**2, 0)[np.newaxis, :]
        D2 = X2 + X2.T - 2*np.dot(X.T, X)
    else:
        X2 = sum(X**2, 0)[:, np.newaxis]
        Y2 = sum(Y**2, 0)[np.newaxis, :]
        D2 = X2 + Y2 - 2*np.dot(X.T, Y)
    return D2

def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if Y.isinstance(bool) and Y is False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T, Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T, Y) + 1
        K = K**kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X, Y)
        K = np.exp(K / (-2 * kernelparameter**2))
    else:
        raise Exception('unspecified kernel')
    return K

class neural_network(torch.nn.Module):
    def __init__(self, layers=[2,100,2], scale=.1, p=0.1, lr=0.1, lam=0.1):
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(scale*torch.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])]
        )
        self.biases = torch.nn.ParameterList(
            [torch.nn.Parameter(scale*torch.randn(n)) for n in layers[1:]]
        )
        self.n_layers = len(layers)
        self.p = p
        self.lr = lr
        self.lam = lam
        self.train = False

    def relu(self, X, W, b):
        out = X @ W + b
        if self.train:
            mask = torch.tensor(
                np.random.binomial(1, 1 - self.p, size=out.shape),
                dtype=out.dtype
            )
            out = mask * (out * (out > 0))
        else:
            out = (1 - self.p) * (X @ W) + b
            out = out * (out > 0)

        return out

    def softmax(self, X, W, b):
        out = X @ W + b
        out = out - out.max(axis=1, keepdims=True).values
        exp_out = out.exp()
        return exp_out / exp_out.sum(axis=1, keepdims=True)

    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float)
        for i in range(self.n_layers - 2):
            X = self.relu(X, self.weights[i], self.biases[i])
        X = self.softmax(X, self.weights[-1], self.biases[-1])
        return X

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        ytrue = ytrue.float()
        return -(ytrue * ypred.log()).sum(axis=1).mean()

    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.lam)

        I = torch.randperm(X.shape[0])
        n = int(np.floor(.9 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            optimizer.zero_grad()
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            output.backward()
            optimizer.step()

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()

def plot_boundary_2d(X, y, model):
    plt.figure(figsize=(6,6))
    plt.scatter(X[y == -1, 0], X[y == -1, 1],
                color='tab:blue', label='Class 1')

    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                color='tab:orange', label='Class 2')

    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    if Z.ndim == 2:
        Z = Z[:, 1] - Z[:, 0]   # decision score for class 2 vs class 1
    Z = Z.reshape(xx.shape)

    if isinstance(model, svm_sklearn) or isinstance(model, svm_qp):
        if isinstance(model, svm_qp):
            sv1 = model.Y_sv == -1
            sv2 = model.Y_sv == 1
        else:
            sv1 = model.y_sv == -1
            sv2 = model.y_sv == 1
        plt.scatter(model.X_sv[sv1, 0], model.X_sv[sv1, 1],
                    marker='x', color='blue', s=80,
                    label='Class 1 SV')

        plt.scatter(model.X_sv[sv2, 0], model.X_sv[sv2, 1],
                    marker='x', color='orange', s=80,
                    label='Class 2 SV')

        plt.contour(xx, yy, Z, levels=[-1, 0, 1],
            colors='k',
            linestyles=['--', '-', '--'])

        plt.legend()
    else:
        plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles=['--'])
    plt.show()

class svm_qp():
    """ Support Vector Machines via Quadratic Programming """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None

    def fit(self, X, Y):

        # INSERT_CODE

        # Here you have to set the matrices as in the general QP problem
        #P =
        #q =
        #G =
        #h =
        #A =   # hint: this has to be a row vector
        #b =   # hint: this has to be a scalar
        n = Y.shape[0]
        K = X @ X.T
        P = np.diag(Y) @ K @ np.diag(Y)
        q = - np.ones((n,1))
        G = np.vstack((- np.eye(n), np.eye(n)))
        h = np.vstack((np.zeros((n,1)), self.C * np.ones((n,1))))
        A = Y[:, None].T
        b = 0

        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()

        eps = 1e-4
        sv = alpha > eps
        self.X_sv = X[sv]
        self.Y_sv = Y[sv]
        self.alpha_sv = alpha[sv]

        self.b = np.mean([
            Y[i] - np.sum(alpha * Y * (X @ X[i]))
            for i in self.Y_sv
        ])

        self.alpha_sv = alpha[sv]

    def predict(self, X):
        K = self.X_sv @ X.T
        return K.T @ (self.alpha_sv * self.Y_sv) + self.b