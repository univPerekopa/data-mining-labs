import numpy as np
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _region_query(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, labels, point_idx, cluster_id):
        neighbors = self._region_query(X, point_idx)
        if len(neighbors) < self.min_samples:
            labels[point_idx] = -1
            return False
        else:
            labels[point_idx] = cluster_id
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                if labels[neighbor_idx] == -1:
                    labels[neighbor_idx] = cluster_id
                elif labels[neighbor_idx] == 0:
                    labels[neighbor_idx] = cluster_id
                    new_neighbors = self._region_query(X, neighbor_idx)
                    if len(new_neighbors) >= self.min_samples:
                        neighbors = np.concatenate((neighbors, new_neighbors))
                i += 1
            return True

    def fit_predict(self, X):
        n_points = X.shape[0]
        labels = np.zeros(n_points, dtype=int) 
        cluster_id = 0

        for point_idx in range(n_points):
            if labels[point_idx] != 0:
                continue
            if self._expand_cluster(X, labels, point_idx, cluster_id + 1):
                cluster_id += 1

        self.labels_ = labels
        return labels - 1


if __name__ == "__main__":
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    db = DBSCAN(eps=0.2, min_samples=5)
    labels = db.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=50)
    plt.title("DBSCAN Clustering")
    plt.show()
