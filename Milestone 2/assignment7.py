from sheet2 import kmeans, em_gmm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

data = np.load('/Users/dmg/Downloads/Problem Set 2-20260520/LabML/Milestone 2/data/5_gaussians.npy').T # (2,500)

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

for k in range(2,8):
    pi, mu, sigma, loglik = em_gmm(data, k)
    plot_gmm_solution(data,mu, sigma, "/Users/dmg/Downloads/Problem Set 2-20260520/LabML/Milestone 2/plots/gmm-"+str(k)+".pdf")

for k in range(2,8):
    mu, labels, loss = kmeans(data, k)
    plot_kmeans(data, mu, labels, "/Users/dmg/Downloads/Problem Set 2-20260520/LabML/Milestone 2/plots/kmeans-"+str(k)+".pdf")