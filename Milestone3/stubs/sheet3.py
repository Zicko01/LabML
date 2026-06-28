import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

def zero_one_loss(y_true, y_pred):
    return np.mean(np.where(y_true != np.sign(y_pred), 1, 0))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def cv(X, y, method, params, loss_function= mean_absolute_error, nfolds=10, nrepetitions=5):
    n, d = X.shape
    cv_loss_min = np.inf
    param_combo = it.product(*params.values())
    n_params = np.prod([len(x) for x in params])
    for itr, param in  enumerate(param_combo):
        if param[0] == "polynomial":
            if not isinstance(param[1], int):
                continue

        if itr == 0:
            start_time = time.perf_counter()
        method_with_param = method(*param)
        error = 0
        for _ in range(nrepetitions):
            idx = np.random.permutation(n)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            X_partitions = np.array_split(X_shuffled, nfolds)
            y_partitions = np.array_split(y_shuffled, nfolds)
            for i in range(nfolds):
                X_train = np.concatenate(X_partitions[:i] + X_partitions[i+1:], axis=0)
                y_train = np.concatenate(y_partitions[:i] + y_partitions[i+1:], axis=0)
                X_test = X_partitions[i]
                y_test = y_partitions[i]
                method_with_param = method_with_param.fit(X_train, y_train)
                y_pred = method_with_param.predict(X_test)
                error += loss_function(y_test, y_pred)

        cv_loss = error / nfolds / nrepetitions

        if n_params == 1:
            return cv_loss

        if cv_loss_min > cv_loss:
            cv_loss_min = cv_loss
            param_best = param

        if itr == 0:
            elapsed_time = time.perf_counter() - start_time
        remaining_time = (n_params - itr) * elapsed_time
        print(f"Validating parameter combination {itr+1} out of {n_params}. Remaining time: {remaining_time:.2f}s")


    method_with_param = method(*param_best).fit(X,y)
    method_with_param.cvloss = cv_loss_min

    return method_with_param

class krr():
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y):
        n, d = X.shape
        self.X_train = X
        # TODO: ask which kernel to use when kernel=Falsex
        if self.kernel == "linear":
            K = X @ X.T
        elif self.kernel == "polynomial":
            K = (X @ X.T + 1) ** self.kernelparameter
        elif self.kernel == "gaussian":
            sqnorms = np.sum(X**2, axis=1)
            sqdist = sqnorms[:, None] + sqnorms[None, :] - 2 * X @ X.T
            K = np.exp(-sqdist / (2 * self.kernelparameter**2))

        if self.regularization == 0:
            L, U = np.linalg.eigh(K)
            eigval_mean = np.mean(L)
            C = (eigval_mean  * np.logspace(-3, 3, 10))
            # use broadcating instead of loop #TODO
            val_loss_min = np.inf
            for c in C:
                # pair-wise multiplication of diagonals in vector form, then convert into diagonal martix
                inv = 1 / (L + c)
                d = np.multiply(L, inv)
                D = np.diag(d)
                Uy = U.T @ y
                Sy =  U @ D @ Uy
                S_ii = (U**2) @ d # faster than U @ d @ U.T
                val_loss = np.mean(np.divide(y - Sy, 1 - S_ii) ** 2)
                if val_loss < val_loss_min:
                    self.alpha = U @ np.diag(inv) @ Uy
                    val_loss_min = val_loss
                    self.regularization = c
        else:
            self.alpha = np.linalg.solve(K + (self.regularization + 1e-4) * np.eye(n), y)
        return self

    def predict(self, X):
        if self.kernel == "linear":
            K = X @ self.X_train.T
        elif self.kernel == "polynomial":
            K = (X @ self.X_train.T + 1) ** self.kernelparameter
        elif self.kernel == "gaussian":
            sqnorms_X = np.sum(X**2, axis=1)
            sqnorms_X_train = np.sum(self.X_train**2, axis=1)
            sqdist = sqnorms_X[:, None] + sqnorms_X_train[None, :] - 2 * X @ self.X_train.T
            K = np.exp(-sqdist / (2 * self.kernelparameter**2))
        return K @ self.alpha