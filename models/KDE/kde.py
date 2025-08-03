import numpy as np
import matplotlib.pyplot as plt
class KDE:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.data = None

    def _kernel_function(self, x):
        """Choose the kernel function based on the kernel type."""
        if self.kernel == 'gaussian':
            return np.exp(-0.5 * (x / self.bandwidth) ** 2) / (np.sqrt(2 * np.pi) * self.bandwidth)
        elif self.kernel == 'box':
            return np.where(np.abs(x) <= self.bandwidth, 0.5 / self.bandwidth, 0)
        elif self.kernel == 'triangular':
            return np.maximum(1 - np.abs(x / self.bandwidth), 0) / self.bandwidth
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, data):
        """Fit KDE with input data."""
        self.data = np.array(data)

    def predict(self, x):
        """Estimate density at a given point x."""
        x = np.atleast_2d(x)
        densities = []
        for xi in x:
            # Calculate density at xi using the selected kernel and bandwidth
            distances = np.linalg.norm(self.data - xi, axis=1)
            kernel_vals = self._kernel_function(distances)
            density = np.sum(kernel_vals) / (len(self.data) * self.bandwidth)
            densities.append(density)
        return np.array(densities)

    def visualize(self, grid_size=50):
        """Plots the 3D density estimation for 2D data with Z as the density."""
        if self.data.shape[1] != 2:
            raise ValueError("Visualization is only supported for 2D data.")
        
        # Create a grid for 3D visualization
        x = np.linspace(self.data[:, 0].min() - 1, self.data[:, 0].max() + 1, grid_size)
        y = np.linspace(self.data[:, 1].min() - 1, self.data[:, 1].max() + 1, grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.c_[X.ravel(), Y.ravel()]

        # Predict densities on the grid
        Z = self.predict(grid_points).reshape(X.shape)

        # 3D plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
        ax.set_xlabel("X axis (feature 1)")
        ax.set_ylabel("Y axis (feature 2)")
        ax.set_zlabel("Density")
        ax.set_title("3D KDE Density Estimation")
        plt.show()
