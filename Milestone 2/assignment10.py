from sheet2 import kmeans, norm_pdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

lab = np.load('data/lab_data.npz')
print(lab.files)
X = lab['X'] # (1000,3)
y = lab['Y'] # (1000,)
print(X.shape, y.shape)


def plot_gmm_solution_3d(X, mu, sigma, labels, path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    k = len(mu)
    colors = plt.cm.tab10.colors

    # Inliers (+1) in green
    inliers = labels == 1
    ax.scatter(
        X[inliers, 0],
        X[inliers, 1],
        X[inliers, 2],
        color="green",
        s=20,
        alpha=0.5,
        label="Inlier (+1)"
    )

    # Outliers (-1) in red
    outliers = labels == -1
    ax.scatter(
        X[outliers, 0],
        X[outliers, 1],
        X[outliers, 2],
        color="red",
        s=20,
        alpha=0.8,
        label="Outlier (-1)"
    )

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)

    sphere = np.array([
        np.cos(u)[None, :] * np.sin(v)[:, None],
        np.sin(u)[None, :] * np.sin(v)[:, None],
        np.cos(v)[:, None] * np.ones_like(u)[None, :]
    ])

    for j in range(k):
        color = colors[j % len(colors)]

        ax.scatter(
            mu[j, 0], mu[j, 1], mu[j, 2],
            color=color,
            marker="x",
            s=100,
            linewidths=2
        )

        eigvals, eigvecs = np.linalg.eigh(sigma[j])

        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        radii = 2 * np.sqrt(eigvals)

        ellipsoid = np.einsum(
            "ij,jkl->ikl",
            eigvecs @ np.diag(radii),
            sphere
        )

        x = ellipsoid[0] + mu[j, 0]
        y = ellipsoid[1] + mu[j, 1]
        z = ellipsoid[2] + mu[j, 2]

        ax.plot_wireframe(
            x, y, z,
            color=color,
            linewidth=1,
            alpha=0.7
        )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path)

def plot__(X, mu, sigma, path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=20, alpha=0.2, color="gray")

    k = len(mu)
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)

    sphere = np.array([
        np.cos(u)[None, :] * np.sin(v)[:, None],
        np.sin(u)[None, :] * np.sin(v)[:, None],
        np.cos(v)[:, None] * np.ones_like(u)[None, :]
    ])

    for j in range(k):
        color = colors[j]

        ax.scatter(
            mu[j, 0], mu[j, 1], mu[j, 2],
            color=color,
            marker="x",
            s=100,
            linewidths=2
        )

        eigvals, eigvecs = np.linalg.eigh(sigma[j])

        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        radii = 2 * np.sqrt(eigvals)

        ellipsoid = np.einsum(
            "ij,jkl->ikl",
            eigvecs @ np.diag(radii),
            sphere
        )

        x = ellipsoid[0] + mu[j, 0]
        y = ellipsoid[1] + mu[j, 1]
        z = ellipsoid[2] + mu[j, 2]

        ax.plot_wireframe(
            x, y, z,
            color=color,
            linewidth=1,
            alpha=0.7
        )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
def plot_only_data(X, path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=X[:, 2],          # color by z-coordinate
        cmap="plasma",
        s=20,
        alpha=0.7
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("x3")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_data(X, mu, sigma, labels, path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    k = len(mu)
    colors = plt.cm.tab10.colors

    # Inliers (+1) in green
    inliers = labels == 1
    ax.scatter(
        X[inliers, 0],
        X[inliers, 1],
        X[inliers, 2],
        color="green",
        s=20,
        alpha=0.5,
        label="Inlier (+1)"
    )

    # Outliers (-1) in red
    outliers = labels == -1
    ax.scatter(
        X[outliers, 0],
        X[outliers, 1],
        X[outliers, 2],
        color="red",
        s=20,
        alpha=0.8,
        label="Outlier (-1)"
    )

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)

    sphere = np.array([
        np.cos(u)[None, :] * np.sin(v)[:, None],
        np.sin(u)[None, :] * np.sin(v)[:, None],
        np.cos(v)[:, None] * np.ones_like(u)[None, :]
    ])

    for j in range(k):
        color = colors[j % len(colors)]

        ax.scatter(
            mu[j, 0], mu[j, 1], mu[j, 2],
            color=color,
            marker="x",
            s=100,
            linewidths=2
        )

        eigvals, eigvecs = np.linalg.eigh(sigma[j])

        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        radii = 2 * np.sqrt(eigvals)

        ellipsoid = np.einsum(
            "ij,jkl->ikl",
            eigvecs @ np.diag(radii),
            sphere
        )

        x = ellipsoid[0] + mu[j, 0]
        y = ellipsoid[1] + mu[j, 1]
        z = ellipsoid[2] + mu[j, 2]

        ax.plot_wireframe(
            x, y, z,
            color=color,
            linewidth=1,
            alpha=0.7
        )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    plt.tight_layout()
    plt.savefig(path)



def gammaidx(X, k):
    #mean distance to the knearest neighbors
    #create k-NN distance matrix for each data point
    diff = X[:, None, :] - X[None, :, :]           # (n, n, d)
    euc_dists = np.sqrt(np.sum(diff**2, axis=(2)))       # (n,n)


    # argpartition is faster than full sort
    # Source: https://www.reddit.com/r/learnpython/comments/bgw7xf/np_argpartition_for_finding_n_minimum_elements_in/
    # diagonals are 0 because distance between itself is 0
    np.fill_diagonal(euc_dists, np.inf)
    idx = np.argpartition(euc_dists, k, axis=1)[:, :k] # (n,k)
    row_idx = np.arange(euc_dists.shape[0])[:, None] # add row indices
    k_distances = euc_dists[row_idx, idx]
    gammas = np.mean(k_distances, axis=1)
    return gammas

def auc(y_true, y_pred, title="Roc Curve", path=None, plot=False):
    o = np.argsort(y_pred)[::-1]
    y_sorted = y_true[o]

    delta_neg = 1 / np.sum(y_true == -1)
    delta_pos = 1 / np.sum(y_true == +1)

    fpr = np.zeros(len(y_true) + 1)
    tpr = np.zeros(len(y_true) + 1)

    for i in range(1, len(y_true) + 1):
        yi = y_sorted[i - 1]

        fpr[i] = fpr[i - 1] + 0.5 * (1 - yi) * delta_neg
        tpr[i] = tpr[i - 1] + 0.5 * (1 + yi) * delta_pos

    auc_val = np.trapz(tpr, fpr)

    if plot:
        plt.figure()
        plt.plot(fpr, tpr, marker='o', label=f'AUC = {auc_val:.3f}')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(path)
        plt.close()

    return auc_val

def gmm_outlier_score(X, pi, mu, sigma):
    n = X.shape[0]
    k = len(pi)

    density = np.zeros(n)

    for j in range(k):
        density += pi[j] * norm_pdf(X, mu[j], sigma[j])

    # low density => likely outlier
    return -np.log(density + 1e-12)

for k in range(1, 11):
    gammas = gammaidx(X, k)
    auc_val = auc(y, gammas, plot=False)
    print(f"AUC for k={k}: {auc_val:.3f}")


def kmeans_best(X, k, n_runs=20, max_iter=100):
    best_loss = np.inf
    best_centers = None
    best_labels = None

    for seed in range(n_runs):
        np.random.seed(seed)

        centers, labels, loss = kmeans(X, k, max_iter=max_iter)

        print(f"Run {seed}: loss = {loss}")

        if loss < best_loss:
            best_loss = loss
            best_centers = centers
            best_labels = labels

    return best_centers, best_labels, best_loss


def em_gmm_this(X, k, max_iter=100, init_kmeans=False, tol=1e-5):
    n, d = X.shape
    pi = np.ones(k) / k
    if init_kmeans:
        mu, _, _ = kmeans_best(X, k, n_runs=20)
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

def plot_gmm_outlier_scores(X, gmm_scores, path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=gmm_scores,
        cmap="plasma",      # low score -> dark, high score -> bright
        s=20,
        alpha=0.8
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("GMM outlier score (-log density)")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


best_loglik = -np.inf
best_result = None

n_runs = 20

for seed in range(n_runs):
    np.random.seed(seed)

    pi_tmp, mu_tmp, sigma_tmp, loglik_tmp = em_gmm_this(
        X,
        3,
        init_kmeans=True
    )

    print(f"Run {seed}: log-likelihood = {loglik_tmp}")

    if loglik_tmp > best_loglik:
        best_loglik = loglik_tmp
        best_result = pi_tmp, mu_tmp, sigma_tmp, loglik_tmp

pi, mu, sigma, loglik = best_result

print("Best log-likelihood:", loglik)
print("Best GMM centers:", mu)

plot_gmm_solution_3d(X, mu, sigma, y, "plots/gmm-lab-best.pdf")
plot__(X, mu, sigma, "plots/gmm-lab-no-labelsb.pdf")
plot_only_data(X, "plots/gmm-lab-data.pdf")
gmm_scores = gmm_outlier_score(X, pi, mu, sigma)
print("-------GMM outlier scores (first 10):", gmm_scores[:10])
plot_gmm_outlier_scores(X, gmm_scores, "plots/gmm-lab-outlier-scores.pdf")
auc_val = auc(y, gmm_scores, title="ROC Curve produced by GMM (k=3) Method", path="plots/lab-gmm-auc.pdf", plot=True)

print("GMM AUC:", auc_val)

for k in range(1, 11):
    gammaidx_scores = gammaidx(X, k)
    auc_val = auc(y, gammaidx_scores, title=f"ROC Curve produced by GammaIdx (k={k}) Method", path=f"plots/lab-gamma-auc-{k}.pdf", plot=True)
    print(f"Gamma AUC for k={k}: {auc_val:.3f}")