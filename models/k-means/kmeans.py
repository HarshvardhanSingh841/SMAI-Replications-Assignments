import numpy as np

class KMeans:
    def __init__(self, k: int, max_iters: int = 1000, tolerance: float = 1e-4):
        """
        Initialize the KMeans clustering model.

        :param k: Number of clusters.
        :param max_iters: Maximum number of iterations for the algorithm.
        :param tolerance: Minimum change in centroids for convergence.
        """
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None

    def fit(self, X: np.ndarray):
        """
        Fit the K-means clustering model to the data by finding optimal centroids.
        
        :param X: The input data of shape (n_samples, n_features).
        """
        # Randomly initialize centroids
        np.random.seed(42)
        initial_centroids_idx = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[initial_centroids_idx]

        for i in range(self.max_iters):
            # Step 1: Assign each point to the nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Step 2: Recompute centroids based on current assignments
            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])

            # Check for convergence (if centroids don't change beyond a threshold)
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break

            self.centroids = new_centroids

    def predict(self, X: np.ndarray):
        """
        Predict the closest cluster each sample in X belongs to.
        
        :param X: The input data of shape (n_samples, n_features).
        :return: Cluster labels for each data point.
        """
        return self._assign_clusters(X)

    #def getCost(self, X: np.ndarray):
        """
        Calculate the Within-Cluster Sum of Squares (WCSS).
        
        :param X: The input data of shape (n_samples, n_features).
        :return: The WCSS value.
        """
        #return np.sum([np.linalg.norm(X[self.labels == j] - self.centroids[j], axis=1) ** 2 for j in range(self.k)])
    
    def getCost(self, X):
        total_cost = 0
        for j in range(self.k):
            points_in_cluster = X[self.labels == j]  # Data points assigned to cluster j
            centroid = self.centroids[j]  # Centroid of cluster j
        
        # Print shapes for debugging
            #print(f"Points in cluster {j}: {points_in_cluster.shape}")
            #print(f"Centroid {j}: {centroid.shape}")
        
        # Ensure that points_in_cluster and centroid are compatible
            if points_in_cluster.shape[0] > 0:  # Only if the cluster has points
                cluster_cost = np.sum(np.linalg.norm(points_in_cluster - centroid, axis=1) ** 2)
                total_cost += cluster_cost
            else:
                print(f"Cluster {j} has no points.")

        return total_cost

    def _assign_clusters(self, X: np.ndarray):
        """
        Assign each data point to the nearest centroid.
        
        :param X: The input data of shape (n_samples, n_features).
        :return: Cluster labels for each data point.
        """
        # Compute the distance from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
