import numpy as np

class PcaAutoencoder:
    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X):
        # Mean center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Calculate eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors
        self.components_ = self.eigenvectors[:, :self.n_components]

    def encode(self, X):
        # Project data onto the lower-dimensional space
        X_centered = X - self.mean
        return np.dot(X_centered, self.components_)

    def forward(self, Z):
        # Reconstruct the data from the reduced representation
        X_reconstructed = np.dot(Z, self.components_.T) + self.mean
        return X_reconstructed