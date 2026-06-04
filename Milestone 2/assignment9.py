from sheet2 import kmeans, em_gmm, kmeans_agglo, agglo_dendro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.io import loadmat
import umap

usps = loadmat('/Users/dmg/Downloads/Problem Set 1-20260414/problem_set1/data/usps.mat')
labels = usps['data_labels'].T # (2007, 10)
patterns = usps['data_patterns'].T # (2007, 256)

def plot_gmm_solution(X, mu, sigma,path):
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

    #ax.set_title("GMM Solution")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    plt.axis('equal')
    plt.savefig(path)

def plot_kmeans(X, centers, labels,path):
    """
    Plot K-means clustering result (for 2D data).

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        Data points.
    centers : ndarray of shape (k, 2)
        Cluster centers.
    labels : ndarray of shape (n_samples,)
        Cluster assignments.
    """
    k = len(centers)

    plt.figure(figsize=(8, 6))

    for i in range(k):
        cluster_points = X[labels == i]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {i}",
            alpha=0.7
        )

    # Plot centers like GMM means
    plt.plot(
        centers[:, 0],
        centers[:, 1],
        'rx',
        markersize=8,
        markeredgewidth=1,
        label="Centers"
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    #plt.title("K-Means Clustering")
    plt.legend()
    plt.savefig(path)

def plot_kmeans_umap(X, centers, labels, path):
    """
    Reduce X and centers to 2D using UMAP, then plot K-means result.
    """

    k = len(centers)

    # Fit UMAP on data + centers so both are in same 2D space
    X_all = np.vstack([X, centers])

    reducer = umap.UMAP(
        n_components=2,
        random_state=0
    )

    X_all_2d = reducer.fit_transform(X_all)

    X_2d = X_all_2d[:len(X)]
    centers_2d = X_all_2d[len(X):]

    plt.figure(figsize=(8, 6))

    for i in range(k):
        cluster_points = X_2d[labels == i]

        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {i}",
            alpha=0.7
        )

    plt.plot(
        centers_2d[:, 0],
        centers_2d[:, 1],
        "rx",
        markersize=8,
        markeredgewidth=1,
        label="Centers"
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

pi, mu, sigma, new_loglik = em_gmm(patterns, k=10)
#centers, nearest, loss = kmeans(patterns, k=10)
#plot_kmeans_umap(patterns, centers, nearest, "plots/ass9-pip install umap-learnkmeans-umap.pdf")