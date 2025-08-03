import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
import os
import sys

# Load your dataset and embeddings
def load_data():
    data_path = os.path.join('..', '..', 'data', 'external', 'word-embeddings.feather')
    df = pd.read_feather(data_path)
    return df

df = load_data()
embeddings = np.array(df.iloc[:, 1].tolist())

# Compute the distance matrix
distance_matrix = pdist(embeddings, metric='euclidean')
distance_matrix2 = pdist(embeddings,metric = 'cityblock')
distance_matrix3 = pdist(embeddings,metric = 'cosine')

# Hierarchical clustering
linkage_methods = ['single', 'complete', 'average', 'ward']
for method in linkage_methods:
    Z = hierarchy.linkage(distance_matrix, method=method)
    #obtaining and printing the linkage matrices
    print(f"Linkage Matrix ({method.capitalize()} Linkage):")
    print(Z)
    print()
    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    hierarchy.dendrogram(Z, leaf_rotation=90., leaf_font_size=12., color_threshold=0.7 * max(Z[:, 2]))
    plt.title(f'Dendrogram ({method.capitalize()} Linkage)')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.show()

# Assume kbest1 and kbest2 are the optimal cluster numbers from K-Means and GMM
kbest1 = 6  # Example value; replace with your result from K-Means
kbest2 = 5  # Example value; replace with your result from GMM

# Create clusters by cutting the dendrogram
def create_clusters(Z, k):
    return hierarchy.fcluster(Z, k, criterion='maxclust')

# Using Euclidean distance metric and best linkage method
best_linkage_method = 'ward'  # Example; replace with your best method based on observations
Z_best = hierarchy.linkage(distance_matrix, method=best_linkage_method)

# Create clusters
clusters_kbest1 = create_clusters(Z_best, kbest1)
clusters_kbest2 = create_clusters(Z_best, kbest2)

# Print cluster assignments
def print_cluster_assignments(clusters, k):
    cluster_dict = {i: [] for i in range(1, k + 1)}
    names = df.iloc[:, 0].values
    results = pd.DataFrame({
    'Word': names,
    'Cluster Assignment': clusters
    })
    # Group words by cluster
    grouped_results = results.groupby('Cluster Assignment')['Word'].apply(list)

    # Print the clusters with their words
    for cluster, words in grouped_results.items():
        print(f"Cluster {cluster}:")
        print(", ".join(words))
    print()  # For better readability between clusters

print("Hierarchical Clustering with kbest1:")
print_cluster_assignments(clusters_kbest1, kbest1)

print("Hierarchical Clustering with kbest2:")
print_cluster_assignments(clusters_kbest2, kbest2)
