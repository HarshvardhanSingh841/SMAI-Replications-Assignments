import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/pca')))
from pca import PCA


# Load your data
def load_data():
    data_path = os.path.join('..', '..', 'data', 'external', 'word-embeddings.feather')
    df = pd.read_feather(data_path)
    return df
data = load_data()
#data = pd.read_feather('word-embeddings.feather')

# Extract embeddings
embeddings = np.array(data.iloc[:, 1].tolist())

# Initialize PCA for 2D
pca_2d = PCA(n_components=2)
pca_2d.fit(embeddings)
embeddings_2d = pca_2d.transform(embeddings)

# Initialize PCA for 3D
pca_3d = PCA(n_components=3)
pca_3d.fit(embeddings)
embeddings_3d = pca_3d.transform(embeddings)

# Check PCA
print("PCA 2D check:", pca_2d.checkPCA(embeddings))
print("PCA 3D check:", pca_3d.checkPCA(embeddings))

# Visualize 2D
plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - 2D Visualization')
plt.show()

# Visualize 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], alpha=0.5)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA - 3D Visualization')
plt.show()
#k2 = 4 or 5 its hard to tell because only 20% of variance is stored in the first 3 principle components.
pca1 = PCA(n_components=min(embeddings.shape))
pca1.fit(embeddings)

#pca2= PCA(n_components = 107)
#pca2.fit(embeddings)
#print("PCA nd check:", pca2.checkPCA(embeddings))

# Plot the explained variance
pca1.plot_explained_variance()