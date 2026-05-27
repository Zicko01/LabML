import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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

def norm_pdf(X, mu, C):
    n, d = X.shape

    eps = 1e-8
    C = C + eps * np.eye(d)

    X_centered = X - mu

    C_inv = np.linalg.solve(C, np.eye(d))
    det_C = np.linalg.det(C)

    exponent = -0.5 * np.sum((X_centered @ C_inv) * X_centered, axis=1)

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
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X[:, 0], X[:, 1], s=20, alpha=0.6)

    k = len(mu)

    for j in range(k):
        ax.plot(
            mu[j, 0],
            mu[j, 1],
            'rx',
            markersize=8,
            markeredgewidth=1
        )

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

    ax.set_title("GMM Solution")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.axis('equal')
    plt.show()

