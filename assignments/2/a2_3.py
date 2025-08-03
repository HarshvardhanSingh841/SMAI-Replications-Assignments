import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/k-means')))
from kmeans import KMeans
# Assuming KMeans class is already defined as per the previous steps.

# Function to load the dataset
def load_data():
    data_path = os.path.join('..', '..', 'data', 'external', 'word-embeddings.feather')
    df = pd.read_feather(data_path)
    return df


# Load the dataset (512-dimensional embeddings)
df = load_data()
print(df.head()) 
X = np.vstack(df['vit'].values)
print("Shape of X:", X.shape)  # Assuming 'embedding' is the 512-dimensional column

def elbow_method(X, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        # Instantiate the KMeans class with k clusters
        kmeans = KMeans(k=k)
        
        # Fit the KMeans model to the data
        kmeans.fit(X)
        
        # Append the Within-Cluster Sum of Squares (WCSS) to the list
        wcss.append(kmeans.getCost(X))
        
        print(f"K: {k}, WCSS: {wcss[-1]}")  # Display WCSS for each value of k
    
    # Plotting the Elbow Graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()

# Call the elbow method with your dataset and a specified range for k
elbow_method(X, max_k=11)  # Adjust max_k based on your needs

# Assume kkmeans1 is the optimal number of clusters determined from the Elbow plot
kkmeans1 = 8  # Replace with your determined value

# Perform K-Means clustering with the optimal number of clusters
kmeans_final = KMeans(k=kkmeans1)
kmeans_final.fit(X)

# Predict clusters for the dataset
clusters = kmeans_final.predict(X)

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