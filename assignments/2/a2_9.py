import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/pca')))
from pca import PCA  

# 1. Load the Spotify dataset and remove non-numerical columns

data_path = os.path.join('..', '..', 'data', 'external', 'spotify.csv')
df = pd.read_csv(data_path)
y = df['track_genre']# Remove non-numerical columns: A (s.no), B (track_id), C (artists), D (album_name), E (track_name), H (explicit), U (track_genre, target variable)
df = df.drop(columns=[ 'track_id', 'artists', 'album_name', 'track_name', 'explicit', 'track_genre'])
df = df.drop(df.columns[0], axis=1)
# Load the dataset
spotify_df = df
print(spotify_df.head())
# Extract features as numpy array
X = spotify_df.values
print (X.shape)
# 2. Standardize the dataset
X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# 3. Apply PCA using your custom PCA class
pca = PCA(n_components=min(X_standardized.shape))  # Fit PCA to all components initially
pca.fit(X_standardized)

# Plot the explained variance to determine optimal number of dimensions
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

#elbow at 6 (near flat after 6th component)
# Choose optimal number of components (adjust this based on scree plot)
optimal_n_components = 6  # Example, adjust after seeing the plot

# Perform PCA with the optimal number of components
pca = PCA(n_components=optimal_n_components)
pca.fit(X_standardized)
reduced_data = pca.transform(X_standardized)

# Print the reduced data shape
print(f"Reduced data shape: {reduced_data.shape}")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/knn')))
from knn import KNN


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets while maintaining the association between samples and labels.
    
    :param X: Features, a numpy array of shape (n_samples, n_features)
    :param y: Labels, a numpy array of shape (n_samples,)
    :param test_size: Proportion of the dataset to include in the test split (default is 0.2)
    :param random_state: Seed for random shuffling (default is None)
    :return: X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Combine features and labels to ensure they remain together during shuffling
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    # Split indices into training and testing sets
    split_idx = int((1 - test_size) * len(indices))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test
print(y.head())

# Create a mapping from category to integer
labels = y.unique()
label_map = {label: idx for idx, label in enumerate(labels)}
print("Label map:", label_map)

y_encoded = y.map(label_map).values

X = reduced_data

# Example usage:
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the KNN model
k = 71  # Choose k based on your experimentation
distance_metric = 'euclidean'  # Choose distance metric

knn = KNN(k=k, distance_metric=distance_metric)

# Fit the KNN model
knn.fit(X_train, y_train)
print("training done") #even though theres no training , just as a time stamp
# Evaluate the KNN model
start_time1 = time.time()
accuracy, precision, recall, f1 = knn.evaluate(X_val, y_val)
end_time1 = time.time()
time_reduced = end_time1 - start_time1 

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
#original data

X = spotify_df.values
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
knn.fit (X_train,y_train)
start_time2 = time.time()
accuracy, precision, recall, f1 = knn.evaluate(X_val, y_val)
end_time2 = time.time()
time_original = end_time2 - start_time2
# Plot the results
labels = ['Original Dataset', 'Reduced Dataset']
times = [time_original, time_reduced]

plt.figure(figsize=(10, 6))
plt.bar(labels, times, color=['blue', 'orange'])
plt.xlabel('Dataset')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time Comparison')
plt.grid(axis='y')
plt.show()

# Print the inference times
print(f"Inference Time (Original Dataset): {time_original:.4f} seconds")
print(f"Inference Time (Reduced Dataset): {time_reduced:.4f} seconds")




