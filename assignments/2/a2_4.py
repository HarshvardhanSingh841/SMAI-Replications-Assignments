import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/gmm')))
from gmm import GMM


# Function to load the dataset
def load_data():
    data_path = os.path.join('..', '..', 'data', 'external', 'word-embeddings.feather')
    df = pd.read_feather(data_path)
    return df

df = load_data()

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

df =load_data()

df.columns = ['word', 'embeddings']

# Convert the 'embeddings' column, which might be stored as an object, into a list or NumPy array
df['embeddings'] = df['embeddings'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)

# Verify the DataFrame structure
print(df.head())
print(f"Shape of DataFrame: {df.shape}")

# Check the shape of the embeddings to ensure they are 512-dimensional
embedding_shape = df['embeddings'].apply(lambda x: len(x)).unique()
print(f"Unique embedding dimensions: {embedding_shape}")

# Load and preprocess your dataset
X = np.array(df['embeddings'].tolist())# Assuming 'vit' contains arrays/lists of features
print(np.isnan(X).sum())  # Count of NaNs
print(np.isinf(X).sum())  # Count of Infs
# Standardize the dataset if necessary
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

def clean_data(X):
    return X[np.isfinite(X).all(axis=1)]

#X_cleaned = clean_data(X_scaled)

# Find the optimal number of clusters for both custom GMM and sklearn GMM
max_k = 10  # Adjust the maximum number of clusters

print(df.head()) 
aic_scores_custom, bic_scores_custom = find_optimal_clusters_custom(X, max_k)
aic_scores_sklearn, bic_scores_sklearn = find_optimal_clusters_sklearn(X, max_k)

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

#kgmm1 = 8? the graph is increasing monotonically so its hard to make out an elbow point.

gmm1 = GMM(k = 8)
#gmm1.fit(X)
#gmm1 = GaussianMixture(n_components= 2 ,covariance_type='full',random_state=42)
gmm1.fit(X)

# Get and print cluster assignments


words = df.iloc[:, 0].values 

cluster_assignments = gmm1.get_cluster_assignments()
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

