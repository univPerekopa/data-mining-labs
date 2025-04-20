import numpy as np
from scipy.spatial.distance import pdist, squareform

class HierarchicalClustering:
    def __init__(self, linkage='single'):
        if linkage not in ['single', 'complete']:
            raise ValueError("linkage must be 'single' or 'complete'")
        self.linkage = linkage

    def fit(self, X):
        self.X = X
        n = len(X)
        self.clusters = [[i] for i in range(n)]
        self.distances = squareform(pdist(X))
        np.fill_diagonal(self.distances, np.inf)
        self.linkage_matrix = []
        current_cluster_idx = n

        cluster_ids = list(range(n))

        while len(self.clusters) > 1:
            i, j = self._find_closest_clusters()
            cluster_i, cluster_j = self.clusters[i], self.clusters[j]
            new_cluster = cluster_i + cluster_j

            dist = self._calculate_distance(cluster_i, cluster_j)

            self.linkage_matrix.append([
                cluster_ids[i],
                cluster_ids[j],
                dist,
                len(new_cluster)
            ])

            self.clusters[i] = new_cluster
            cluster_ids[i] = current_cluster_idx
            del self.clusters[j]
            del cluster_ids[j]
            current_cluster_idx += 1

        return np.array(self.linkage_matrix)

    def _find_closest_clusters(self):
        min_dist = np.inf
        closest = (0, 1)
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                dist = self._calculate_distance(self.clusters[i], self.clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    closest = (i, j)
        return closest

    def _calculate_distance(self, cluster1, cluster2):
        distances = [self.distances[i, j] for i in cluster1 for j in cluster2]
        return min(distances) if self.linkage == 'single' else max(distances)

# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import dendrogram

# # Generate sample data
# X, _ = make_blobs(n_samples=10, centers=3, random_state=42)

# # Perform clustering
# hc = HierarchicalClustering(linkage='complete')  # Try 'single' or 'complete'
# Z = hc.fit(X)
# print(Z)

# # Plot dendrogram
# dendrogram(Z)
# plt.title("Hierarchical Clustering Dendrogram")
# plt.xlabel("Sample index")
# plt.ylabel("Distance")
# plt.show()
