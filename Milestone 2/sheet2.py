import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram

def kmeans(X, k, max_iter=100):
    n, d = X.shape

    centers = X[np.random.choice(n, k, replace=False)]

    for i in range(max_iter):
        distances = np.sum((X[:, None] - centers) ** 2, axis=2)
        nearest = np.argmin(distances, axis=1)

        new_centers = np.array([X[nearest == j].mean(axis=0) for j in range(k)])

        number_of_changes = np.sum(centers != new_centers)
        loss = np.sum((X - new_centers[nearest]) ** 2)

        print(f"Iteration {i + 1}, number of changes: {number_of_changes}, loss: {loss}")
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return centers, nearest, loss

def kmeans_loss(X, r):
    loss = 0.0

    for c in np.unique(r):
        points = X[r == c]

        centroid = np.mean(points, axis=0)
        loss += np.sum((points - centroid) ** 2)

    return loss

def kmeans_agglo(X, r):
    clusters = np.unique(r)
    k = len(clusters)
    n = len(r)

    if k < 2:
        raise ValueError(
            f"Need at least 2 clusters, got {k}."
        )

    R = np.zeros((k - 1, n), dtype=int)
    kmloss = np.zeros(k)
    mergeidx = np.zeros((k - 1, 2), dtype=int)

    R[0] = r
    kmloss[0] = kmeans_loss(X, r)

    current_r = r.copy()

    for step in range(k - 1):

        current_clusters = np.unique(current_r)

        best_loss = np.inf
        best_r = None
        best_pair = None

        for i in range(len(current_clusters)):
            for j in range(i + 1, len(current_clusters)):

                c1 = current_clusters[i]
                c2 = current_clusters[j]

                trial_r = current_r.copy()

                # merge c2 into c1
                new_idx = max(trial_r) + 1
                trial_r[trial_r == c1] = new_idx
                trial_r[trial_r == c2] = new_idx

                loss = kmeans_loss(X, trial_r)

                if loss < best_loss:
                    best_loss = loss
                    best_r = trial_r
                    best_pair = (c1, c2)

        mergeidx[step] = best_pair
        kmloss[step + 1] = best_loss

        current_r = best_r

        if step + 1 < k - 1:
            R[step + 1] = current_r

    return R, kmloss, mergeidx

def agglo_dendro(kmloss, mergeidx):
    m = mergeidx.shape[0]
    k = m + 1

    Z = np.zeros((m, 4))

    sizes = {i: 1 for i in range(k)}

    for i in range(m):
        c1, c2 = mergeidx[i]

        Z[i, 0] = c1
        Z[i, 1] = c2
        Z[i, 2] = kmloss[i + 1] - kmloss[i]

        new_cluster = k + i

        size1 = sizes[c1]
        size2 = sizes[c2]

        Z[i, 3] = size1 + size2

        sizes[new_cluster] = size1 + size2

    plt.figure()
    dendrogram(Z)
    plt.xlabel("Cluster index")
    plt.ylabel("Increase in criterion function")
    plt.title("Hierarchical cluster dendrogram")
    plt.show()

def norm_pdf(X, mu, C):
    n, d = X.shape

    eps = 1e-4
    C = C + eps * np.eye(d)

    X_centered = X - mu

    C_inv = np.linalg.solve(C, np.eye(d))
    det_C = np.linalg.det(C)
    det_C = max(det_C, 1e-300)

    exponent = -0.5 * np.sum((X_centered @ C_inv) * X_centered, axis=1)
    exponent = np.maximum(exponent, -700)

    const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_C)

    y = const * np.exp(exponent)

    return y

def em_gmm(X, k, max_iter=100, init_kmeans=False, tol=1e-5):
    n, d = X.shape
    pi = np.ones(k) / k
    if init_kmeans:
        mu, _, _ = kmeans(X, k)
    else:
        mu = X[np.random.choice(n, k, replace=False)]

    sigma = np.array([np.eye(d) for _ in range(k)])

    loglik = -np.inf

    for i in range(max_iter):
        gamma = np.zeros((n, k))

        for j in range(k):
            gamma[:, j] = pi[j] * norm_pdf(X, mu[j], sigma[j])
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma = gamma / gamma_sum

        Nk = np.sum(gamma, axis=0)
        pi = Nk / n

        for j in range(k):
            if Nk[j] < 1e-10:
                continue

            mu[j] = np.sum(gamma[:, j][:, None] * X, axis=0) / Nk[j]
            X_centered = X - mu[j]
            sigma[j] = (gamma[:, j][:, None] * X_centered).T @ X_centered / Nk[j]

        new_loglik = np.sum(np.log(gamma_sum.squeeze()))

        print(f"Iteration {i + 1}, log-likelihood: {new_loglik:.6f}")

        if np.abs(new_loglik - loglik) < tol:
            print("Convergence reached.")
            break

        loglik = new_loglik

    return pi, mu, sigma, new_loglik

def plot_gmm_solution(X, mu, sigma):
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
    

    k = len(mu)

    for j in range(k):
        ax.scatter(mu[j, 0], mu[j, 1], c='red', marker='x', s=100)

        eigvals, eigvecs = np.linalg.eigh(sigma[j])

        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        angle = np.degrees(
            np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        )

        width = 2 * 2 * np.sqrt(eigvals[0])
        height = 2 * 2 * np.sqrt(eigvals[1])

        ellipse = Ellipse(
            xy=mu[j],
            width=width,
            height=height,
            angle=angle,
            edgecolor='red',
            facecolor='none',
            linewidth=1
        )

        ax.add_patch(ellipse)

    ax.set_title("GMM")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.axis('equal')
    plt.show()

