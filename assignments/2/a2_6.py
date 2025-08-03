import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os
import sys



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/k-means')))
from kmeans import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/pca')))
from pca import PCA




# Assuming 4 to be k2 from the 2d data visualisation
k2 = 4
def load_data():
    data_path = os.path.join('..', '..', 'data', 'external', 'word-embeddings.feather')
    df = pd.read_feather(data_path)
    return df


# Load the dataset (512-dimensional embeddings)
df = load_data()
print(df.head()) 
X = np.vstack(df['vit'].values)
print("Shape of X:", X.shape)  # Assuming 'embedding' is the 512-dimensional column
# Initialize and fit K-means
kmeans = KMeans(k=k2)
kmeans.fit(X)

# Get cluster assignments
#kmeans_labels = kmeans.predict(X)

# Optionally, print the cluster assignments
#print(f"K-means cluster assignments (2D): {kmeans_labels_2D}")
# Predict clusters for the dataset
clusters = kmeans.predict(X)

names = df.iloc[:, 0].values
# You can now use `clusters` for further analysis, plotting, etc.
#print(f"Final cluster assignments for each data point: {clusters}")

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


import matplotlib.pyplot as plt

embeddings = np.array(df.iloc[:, 1].tolist())

# Assuming you have PCA class implemented as pca
pca = PCA(n_components=min(embeddings.shape))
pca.fit(embeddings)

# Plot the explained variance ratio
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

#because thats when 90% reconstruction threshold is achieved
#i,e check pca becomes true at 107 principle components
#but from the scree plot the plot flattens after 10ish components so reduced dims are taken to be 10.
optimal_n_components = 10  # Example value; adjust based on scree plot

# Perform PCA
pca = PCA(n_components=optimal_n_components)
pca.fit(embeddings)
pca.checkPCA(embeddings)
reduced_embeddings = pca.transform(embeddings)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for k in range(1, 11):  # Test k from 1 to 10
    kmeans = KMeans(k=k)
    kmeans.fit(reduced_embeddings)
    wcss.append(kmeans.getCost(reduced_embeddings))

# Plot the Elbow Method results
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

#Choose the optimal number of clusters from the plot
kkmeans3 = 6  

#Perform K-means clustering on reduced dataset
kmeans = KMeans(k=kkmeans3)
kmeans.fit(reduced_embeddings)

# Get cluster assignments
kmeans_labels_reduced = kmeans.predict(reduced_embeddings)

# Optionally, print the cluster assignments
#print(f"K-means cluster assignments (reduced): {kmeans_labels_reduced}")

clusters = kmeans.predict(reduced_embeddings)

names = df.iloc[:, 0].values
# You can now use `clusters` for further analysis, plotting, etc.
print(f"Final cluster assignments for each data point: {clusters}")

# Create a dictionary to store cluster assignments
cluster_dict = {i: [] for i in range(kkmeans3)}

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


#6.3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/gmm')))
from gmm import GMM



df =load_data()

df.columns = ['word', 'embeddings']

# Convert the 'embeddings' column, which might be stored as an object, into a list or NumPy array
df['embeddings'] = df['embeddings'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)


X = np.array(df['embeddings'].tolist())

k2 = 4
gmm2 = GMM(k2)
gmm2.fit(X)


words = df.iloc[:, 0].values 

cluster_assignments = gmm2.get_cluster_assignments()
#print("Cluster Assignments:\n", cluster_assignments)
results = pd.DataFrame({
    'Word': words,
    'Cluster Assignment': cluster_assignments
})
# Group words by cluster
grouped_results = results.groupby('Cluster Assignment')['Word'].apply(list)

# Print the clusters with their words
for cluster, words in grouped_results.items():
    print(f"Cluster {cluster}:")
    print(", ".join(words))
    print()  # For better readability between clusters

#PCA+GMM 6.4

def compute_aic_bic(gmm, X):
    # Number of data points
    n = X.shape[0]
    
    # Compute log-likelihood
    log_likelihood = gmm.getLikelihood(X)
    
    # Number of parameters
    k = gmm.k
    d = X.shape[1]
    
    # Calculate number of parameters
    num_params = k * (d + (d * (d + 1)) / 2 + 1)
    
    # Compute AIC and BIC
    aic = 2 * num_params - 2 * log_likelihood
    bic = np.log(n) * num_params - 2 * log_likelihood
    
    return aic, bic

def find_optimal_clusters_custom(X, max_k):
    aic_scores = []
    bic_scores = []
    
    for k in range(1, max_k + 1):
        gmm = GMM(k=k)
        gmm.fit(X)
        aic, bic = compute_aic_bic(gmm, X)
        aic_scores.append(aic)
        bic_scores.append(bic)
    
    return aic_scores, bic_scores

def find_optimal_clusters_sklearn(X, max_k):
    aic_scores = []
    bic_scores = []
    
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(X)
        aic_scores.append(gmm.aic(X))
        bic_scores.append(gmm.bic(X))
    
    return aic_scores, bic_scores



max_k = 10  # Adjust the maximum number of clusters

print(df.head()) 
aic_scores_custom, bic_scores_custom = find_optimal_clusters_custom(reduced_embeddings, max_k)
aic_scores_sklearn, bic_scores_sklearn = find_optimal_clusters_sklearn(reduced_embeddings, max_k)

# Plot AIC and BIC scores for both implementations
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, max_k + 1), aic_scores_custom, label='Custom GMM AIC', marker='o')
plt.plot(range(1, max_k + 1), bic_scores_custom, label='Custom GMM BIC', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Custom GMM AIC and BIC')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, max_k + 1), aic_scores_sklearn, label='sklearn GMM AIC', marker='o')
plt.plot(range(1, max_k + 1), bic_scores_sklearn, label='sklearn GMM BIC', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('sklearn GMM AIC and BIC')
plt.legend()

plt.tight_layout()
plt.show()


#strong elbow on 7 in sklearn and slight elbow at 5 in my custom
kgmm3 = 5
gmm3 = GMM(kgmm3)
gmm3.fit(reduced_embeddings)

cluster_assignments = gmm3.get_cluster_assignments()
#print("Cluster Assignments:\n", cluster_assignments)
results = pd.DataFrame({
    'Word': names,
    'Cluster Assignment': cluster_assignments
})
# Group words by cluster
grouped_results = results.groupby('Cluster Assignment')['Word'].apply(list)

# Print the clusters with their words
for cluster, words in grouped_results.items():
    print(f"Cluster {cluster}:")
    print(", ".join(words))
    print()  # For better readability between clusters
