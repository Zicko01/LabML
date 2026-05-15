import numpy as np
import matplotlib.pyplot as plt

class PCA():
    """
    PCA implementation.

    Attributes:
        C (np.ndarray): Mean vector of shape (d,).
        U (np.ndarray): Principal directions of shape (d, d), where each
            column represents a principal component.
        D (np.ndarray): Principal values of shape (d,), sorted in
            descending order.
    """

    def __init__(self, Xtrain):
        """
        Initializes PCA components from the training data.

        Args:
            Xtrain (np.ndarray): Training dataset of shape (n, d),
                where n is the number of samples and d is the number
                of features.
        """
        self.C = np.mean(Xtrain, axis=0)

        N = Xtrain.shape[0]
        Z = Xtrain - self.C

        U, S, _ = np.linalg.svd(Z.T)

        self.U = U
        self.D = S ** 2 / (N - 1)

    def project(self, Xtest, m):
        """
        Projects test data onto the first m principal components.

        The method centers the data using the training mean and
        projects it into the m-dimensional PCA subspace.

        Returns a matrix Z of shape (n, m), where n is the number
        of test samples.

        Args:
            Xtest (np.ndarray): Test dataset of shape (n, d).
            m (int): Number of principal components to use.

        Returns:
            np.ndarray: Projected data of shape (n, m).
        """
        U_m = self.U[:, :m]
        Z = (Xtest - self.C) @ U_m

        return Z

    def denoise(self, Xtest, m):
        """
        Reconstructs (denoises) the input data using the first m
        principal components.

        The method projects the data into the m-dimensional PCA space
        and reconstructs it back into the original d-dimensional space.

        Returns a matrix Y of shape (n, d), where n is the number
        of test samples.

        Args:
            Xtest (np.ndarray): Test dataset of shape (n, d).
            m (int): Number of principal components to use.

        Returns:
            np.ndarray: Reconstructed (denoised) data of shape (n, d).
        """
        Z = self.project(Xtest, m)
        U_m = self.U[:, :m]

        Y = Z @ U_m.T + self.C
        return Y
    
def gammaidx(X, k):
    """
    Compute the gamma index for each sample in X with respect to the k nearest neighbors.

    Parameters:
    X (numpy.ndarray): The input data of shape (n, d).
    k (int): The number of nearest neighbors to consider.

    Returns:
    numpy.ndarray: An array of gamma indices for each sample.
    """
    n = X.shape[0]
    y = np.zeros(n)

    for i in range(n):
        dist = np.sum((X - X[i]) ** 2, axis = 1)
        closest = np.argsort(dist)[1:k+1]
        y[i] = np.mean(np.sqrt(dist[closest]))

    return y

def auc(y_true, y_pred, plot=False):
    o = np.argsort(-y_pred)
    d_neg = 1 / np.sum(y_true == -1)
    d_pos = 1 / np.sum(y_true == 1)

    y_sorted = y_true[o]

    fpr = [0.0]
    tpr = [0.0]

    for y in y_sorted:
        fpr.append(fpr[-1] + 0.5 * (1 - y) * d_neg)
        tpr.append(tpr[-1] + 0.5 * (1 + y) * d_pos)

    fpr = np.array(fpr)
    tpr = np.array(tpr)

    c = np.trapezoid(tpr, fpr)

    if plot:
        plt.plot(fpr, tpr, marker='o')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.grid()
        plt.show()

    return c

def lle(X, m, tol, n_rule, k=None, epsilon=None):
    N = X.shape[0]
    W = np.zeros((N, N))

    if n_rule == 'knn':
        if k is None:
            raise ValueError('k nearest neighbours not specified!')
        if k >= N:
            raise ValueError('k must be < number of points')
        if k <= 0:
            raise ValueError('k must be a positive number!')
    elif n_rule == 'eps-ball':
        if epsilon is None:
            raise ValueError('epsilon not specified!')
        if epsilon <= 0:
            raise ValueError('epsilon must be a positive number!')
    else:
        raise ValueError('n_rule can be either \'knn\' or \'eps-ball\'!')       


    for i in range(N):
        dist = np.sum((X - X[i]) ** 2, axis = 1)

        if n_rule == 'knn':
            closest = np.argsort(dist)[1:k+1]
            ind = np.zeros_like(dist, dtype=bool)
            ind[closest] = True
        else:
            ind = dist < epsilon ** 2
            ind[i] = False

            if not np.any(ind):
                raise ValueError("No neighbors found for this epsilon!")

        Z = X[ind] - X[i]
        C = Z @ Z.T
        ni = C.shape[0]

        C += tol * np.eye(ni)

        ones = np.ones(ni)
        w = np.linalg.solve(C, ones)
        w = w / np.sum(w)
        # --------------------------------------
        W[i,ind] = w

    M = (np.eye(N) - W).T @ (np.eye(N) - W)

    _, eigvecs = np.linalg.eigh(M)

    Y = eigvecs[:, 1:m+1]

    return Y

