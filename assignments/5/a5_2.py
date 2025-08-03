import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/KDE')))
from kde import KDE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/gmm')))
from gmm import GMM


def generate_synthetic_data():
    # Large diffused circle (3000 points)
    large_radius = 2.2
    large_points = 3000
    large_angles = np.random.uniform(0, 2 * np.pi, large_points)
    large_radii = large_radius * np.sqrt(np.random.uniform(0, 1, large_points))  # Uniform distribution in circle
    large_x = large_radii * np.cos(large_angles) + np.random.normal(0, 0.2, large_points)
    large_y = large_radii * np.sin(large_angles) + np.random.normal(0, 0.2, large_points)

    # Smaller dense circle (500 points)
    small_radius = 0.25
    small_points = 500
    small_angles = np.random.uniform(0, 2 * np.pi, small_points)
    small_radii = small_radius * np.sqrt(np.random.uniform(0, 1, small_points))
    small_x = 1 + small_radii * np.cos(small_angles) + np.random.normal(0, 0.05, small_points)  # Offset in x
    small_y = 1 + small_radii * np.sin(small_angles) + np.random.normal(0, 0.05, small_points)  # Offset in y

    # Combine both circles
    x = np.concatenate([large_x, small_x])
    y = np.concatenate([large_y, small_y])

    # Plot the synthetic data
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, color='black')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title("Synthetic Data")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

    return np.vstack([x, y]).T

# Generate and visualize synthetic data
data = generate_synthetic_data()


kde = KDE(kernel='gaussian', bandwidth=0.3)  # Adjust bandwidth as necessary
kde.fit(data)

# Fit GMM model with two components
gmm_2_components = GMM(k=2)
gmm_2_components.fit(data)

# Define a grid for visualizing KDE and GMM density
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
grid = np.c_[X.ravel(), Y.ravel()]

# KDE density estimation
kde_density = kde.predict(grid).reshape(X.shape)

# GMM density estimation for 2 components
gmm_density_2 = np.sum([
    gmm_2_components.weights[i] * multivariate_normal(mean=gmm_2_components.means[i], cov=gmm_2_components.covariances[i]).pdf(grid)
    for i in range(2)
], axis=0).reshape(X.shape)

# Plotting KDE in 3D
fig = plt.figure(figsize=(14, 6))

# KDE 3D plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, kde_density, cmap='viridis', edgecolor='k', alpha=0.8)
ax1.set_title("3D KDE Density Estimation")
ax1.set_xlabel("X axis (feature 1)")
ax1.set_ylabel("Y axis (feature 2)")
ax1.set_zlabel("Density")

# GMM 3D plot (2 components)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, gmm_density_2, cmap='viridis', edgecolor='k', alpha=0.8)
ax2.set_title("3D GMM Density Estimation (2 Components)")
ax2.set_xlabel("X axis (feature 1)")
ax2.set_ylabel("Y axis (feature 2)")
ax2.set_zlabel("Density")

plt.tight_layout()
plt.show()


#I could have used kde.visualise() also but it causes the things to not be showable side by side in a subplot 
kde.visualize()


gmm_5_components = GMM(k=5)
gmm_5_components.fit(data)

gmm_density_5 = np.sum([
    gmm_5_components.weights[i] * multivariate_normal(mean=gmm_5_components.means[i], cov=gmm_5_components.covariances[i]).pdf(grid)
    for i in range(5)
], axis=0).reshape(X.shape)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, gmm_density_5, cmap='viridis', edgecolor='k', alpha=0.8)
ax.set_title("3D GMM Density Estimation (5 Components)")
ax.set_xlabel("X axis (feature 1)")
ax.set_ylabel("Y axis (feature 2)")
ax.set_zlabel("Density")

plt.show()