import numpy as np
import matplotlib.pyplot as plt

class KMedoids:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.medoids = None

    def _initialize_medoids(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return indices

    def _compute_distance_matrix(self, X):
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            dist_matrix[i] = np.linalg.norm(X - X[i], axis=1)
        return dist_matrix

    def _assign_clusters(self, dist_matrix, medoid_indices):
        return np.argmin(dist_matrix[:, medoid_indices], axis=1)

    def _update_medoids(self, X, labels):
        new_medoids = []
        for i in range(self.n_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                new_medoids.append(np.random.choice(range(len(X))))
                continue
            intra_distances = np.sum(
                np.linalg.norm(X[cluster_indices][:, np.newaxis] - X[cluster_indices], axis=2), axis=1
            )
            min_index = cluster_indices[np.argmin(intra_distances)]
            new_medoids.append(min_index)
        return np.array(new_medoids)

    def fit_predict(self, X):
        dist_matrix = self._compute_distance_matrix(X)
        medoid_indices = self._initialize_medoids(X)

        for _ in range(self.max_iters):
            labels = self._assign_clusters(dist_matrix, medoid_indices)
            new_medoid_indices = self._update_medoids(X, labels)
            if np.all(medoid_indices == new_medoid_indices):
                break
            medoid_indices = new_medoid_indices

        self.medoids = X[medoid_indices]
        return labels


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

    kmedoids = KMedoids(n_clusters=4, random_state=42)
    labels = kmedoids.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(kmedoids.medoids[:, 0], kmedoids.medoids[:, 1], c='red', s=200, marker='D', label="Medoids")
    plt.legend()
    plt.title("K-Medoids Clustering")
    plt.show()
