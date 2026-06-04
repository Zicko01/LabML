from sheet2 import kmeans, em_gmm, norm_pdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

lab = np.load('data/lab_data.npz')
print(lab.files)
X = lab['X'] # (1000,3)
y = lab['Y'] # (1000,)
print(X.shape, y.shape)

pi, mu, sigma, new_loglik = em_gmm(X, 3, init_kmeans=True)
print("GMM log-likelihood:", new_loglik)
print("GMM centers:", mu)

def plot_gmm_solution_3d(X, mu, sigma, path):
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
            mu[j, 0],
            mu[j, 1],
            mu[j, 2],
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


plot_gmm_solution_3d(X, mu, sigma, "plots/gmm-lab.pdf")

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

def auc(y_true, y_pred, plot=False):
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
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

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

gmm_scores = gmm_outlier_score(X, pi, mu, sigma)

auc_val = auc(y, gmm_scores)

print("GMM AUC:", auc_val)