import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.threshold = 0.9

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvalues and eigenvectors
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Select top n_components
        self.components = eigenvectors[:, :self.n_components]

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_variances = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = explained_variances / total_variance
        
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    def inverse_transform(self, X_reduced):
        return np.dot(X_reduced, self.components.T) + self.mean

    def checkPCA(self, X):
        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)
        
        # Compute reconstruction error (mean squared error)
        reconstruction_error = np.mean(np.square(X - X_reconstructed))
        
        # Compute total variance of the original data
        total_variance = np.mean(np.square(X - np.mean(X, axis=0)))
        
        # Compute ratio of retained variance
        retained_variance_ratio = 1 - (reconstruction_error / total_variance)
        
        return retained_variance_ratio >= self.threshold
    #def checkPCA(self, X):
        #transformed = self.transform(X)
        #return transformed.shape[1] == self.n_components
    
    def plot_explained_variance(self):
        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        
        # Plot cumulative explained variance
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(cumulative_variance, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by PCA Components')
        plt.grid(True)
        plt.show()
