import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
class GMM:
    def __init__(self, k, tol=1e-6, max_iter=100):
        self.k = k  # Number of clusters
        self.tol = tol  # Tolerance for convergence
        self.max_iter = max_iter  # Maximum number of iterations

    def _initialize_parameters(self, X):
        """ Initialize GMM parameters (means, covariances, weights, responsibilities) """
        n, d = X.shape
        self.means = X[np.random.choice(n, self.k, False)]  # Randomly select initial means
        # Regularize covariance matrices by adding a small value to the diagonal
        self.covariances = np.array([np.cov(X.T) + 1e-2 * np.eye(d) for _ in range(self.k)])
        self.weights = np.ones(self.k) / self.k  # Initialize equal weights
        self.responsibilities = np.zeros((n, self.k))  # Initialize responsibilities

    def fit(self, X):
        """ Fit GMM to the data using EM algorithm """
        self._initialize_parameters(X)
        n, d = X.shape

        for iteration in range(self.max_iter):
            # E-Step: Compute responsibilities (soft assignments)
            for i in range(self.k):
                # Add regularization to covariance matrix to avoid singularities
                reg_cov = self.covariances[i] + 1e-2 * np.eye(d)
                rv = multivariate_normal(mean=self.means[i], cov=reg_cov)
                self.responsibilities[:, i] = self.weights[i] * rv.pdf(X)

            # Normalize the responsibilities to avoid division by zero
            responsibilities_sum = self.responsibilities.sum(axis=1)[:, np.newaxis]
            responsibilities_sum = np.maximum(responsibilities_sum, 1e-10)  # Prevent div by 0
            self.responsibilities = self.responsibilities / responsibilities_sum

            # M-Step: Update GMM parameters
            Nk = self.responsibilities.sum(axis=0)

            # Update means
            self.means = np.dot(self.responsibilities.T, X) / Nk[:, None]

            # Update covariances with regularization
            self.covariances = np.array([
                np.dot(
                    (self.responsibilities[:, j] * (X - self.means[j]).T), 
                    (X - self.means[j])
                ) / Nk[j] + 1e-2 * np.eye(d)  # Regularize covariance
                for j in range(self.k)
            ])

            # Update weights
            self.weights = Nk / n

            # Compute the log-likelihood and check for convergence
            log_likelihood = self.getLikelihood(X)
            if iteration > 0 and np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break  # Stop if the change in log-likelihood is smaller than tolerance
            prev_log_likelihood = log_likelihood

    def getParams(self):
        """ Get the current GMM parameters """
        return {
            'means': self.means,
            'covariances': self.covariances,
            'weights': self.weights
        }

    def getMembership(self):
        """ Get the current membership (responsibilities) for each data point """
        return self.responsibilities

    def getLikelihood(self, X):
        """ Compute the log-likelihood of the data given the current GMM parameters """
        likelihood = np.sum([
            self.weights[i] * multivariate_normal(mean=self.means[i], cov=self.covariances[i]).pdf(X)
            for i in range(self.k)
        ], axis=0)
        # Clip likelihood values to avoid log(0) and overflow errors
        likelihood = np.clip(likelihood, 1e-10, None)
        return np.sum(np.log(likelihood))
    
    def get_cluster_assignments(self):
        """ Determine the cluster assignment for each data point """
        cluster_assignments = np.argmax(self.responsibilities, axis=1)
        return cluster_assignments




def plot_gmm_clusters(X, gmm):
    # Plot data points
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c='gray', s=10, label='Data points')
    
    # Plot GMM means
    means = gmm.getParams()['means']
    plt.scatter(means[:, 0], means[:, 1], c='red', s=100, marker='x', label='Cluster Means')

    # Plot ellipses representing covariance matrices
    covariances = gmm.getParams()['covariances']
    weights = gmm.getParams()['weights']
    
    for i in range(len(means)):
        cov = covariances[i]
        v, w = np.linalg.eigh(cov)  # Eigenvalues and eigenvectors
        order = v.argsort()[::-1]  # Sort eigenvalues and corresponding eigenvectors
        v = v[order]
        w = w[:, order]
        
        # Width and height of the ellipse
        width, height = 2 * np.sqrt(v)
        angle = np.degrees(np.arctan2(*w[:, 0][::-1]))  # Compute angle from eigenvector

        # Draw the ellipse
        ell = Ellipse(xy=means[i], width=width, height=height, angle=angle, 
                      color='blue', alpha=0.5, label=f'Cluster {i+1} Covariance')
        plt.gca().add_patch(ell)

    plt.title('GMM Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    #Generate synthetic data
    X, _ = make_blobs(n_samples=500, centers=5, n_features=2, random_state=42)

     #Initialize and fit GMM
    gmm = GMM(k=5)
    gmm.fit(X)

    # Print GMM parameters
    #print("Means:\n", gmm.getParams()['means'])
    #print("Covariances:\n", gmm.getParams()['covariances'])
    #print("Weights:\n", gmm.getParams()['weights'])

    # Get and print membership probabilities
    #membership = gmm.getMembership()
    #print("Membership:\n", membership)

    # Print likelihood
    print("Likelihood:", gmm.getLikelihood(X))
    plot_gmm_clusters(X, gmm)