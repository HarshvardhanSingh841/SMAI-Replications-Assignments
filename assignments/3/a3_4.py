import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import wandb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/Autoencoders')))
from AutoEncoder import AutoEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/knn')))
from knn import KNN



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
# Step 1: Normalize the data (scale to [0, 1] range)
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(X)

# Step 2: Standardize the normalized data (zero mean, unit variance)
scaler = StandardScaler()
X = normalized_data

print(X.shape[1])
X_train, X_val, y_train , y_val = train_test_split(X,y, test_size=0.2, random_state=42)
# 1. Fit KNN on Raw Dataset
knn_raw = KNeighborsClassifier(n_neighbors=5)  # You can choose a different number of neighbors
knn_raw.fit(X_train, y_train)
y_pred_raw = knn_raw.predict(X_val)

# Calculate metrics for the raw dataset
accuracy_raw = accuracy_score(y_val, y_pred_raw)
precision_raw = precision_score(y_val, y_pred_raw, average='weighted')
recall_raw = recall_score(y_val, y_pred_raw, average='weighted')
f1_raw = f1_score(y_val, y_pred_raw, average='weighted')

print("Metrics for Raw Dataset:")
print(f"Accuracy: {accuracy_raw:.4f}")
print(f"Precision: {precision_raw:.4f}")
print(f"Recall: {recall_raw:.4f}")
print(f"F1 Score: {f1_raw:.4f}")

# # 3. Define the AutoEncoder parameters
# input_size = X.shape[1]  # Number of input features
# latent_size = 6  # Reduced dimensions (based on your PCA result)

# # 4. Initialize the AutoEncoder
# autoencoder = AutoEncoder(input_size=input_size, latent_size=latent_size, learning_rate=0.001, epochs=500)

# # 5. Train the AutoEncoder
# autoencoder.fit(X)

# # 6. Get the latent (reduced) representation of the dataset
# latent_data = autoencoder.get_latent(X)
# print("Latent data shape (reduced dimensions):", latent_data.shape)

# # 7. Optionally, you can visualize the latent space
# plt.scatter(latent_data[:, 0], latent_data[:, 1], c=y, cmap='viridis', alpha=0.7)
# plt.title('2D Visualization of Latent Space')
# plt.xlabel('Latent Dimension 1')
# plt.ylabel('Latent Dimension 2')
# plt.colorbar(label='Track Genre')
# plt.show()

# # 8. Optionally, you can reconstruct the original data from the latent space
# reconstructed_data = autoencoder.reconstruct(X)
# print("Reconstructed data shape:", reconstructed_data.shape)



# 4. Instantiate the AutoEncoder


# 5. Split the data into training and validation sets
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
# Example usage of the AutoEncoder class
input_size = X.shape[1]  # Number of features in your dataset
hidden_size = 10  # Number of neurons in the hidden layers
latent_size = 6  # Reduced dimensions (latent space)

print(X_train)

# Initialize the AutoEncoder
autoencoder = AutoEncoder(input_size, hidden_size, latent_size, learning_rate = 0.00001, epochs = 100)

# Train on your data (X)
autoencoder.fit(X)

# Get the latent space representation
latent_representation = autoencoder.get_latent(X)
print(latent_representation)
# Reconstruct the input from the latent space
reconstructed_data = autoencoder.reconstruct(X)
# 9. Use the KNN classifier on the latent representation (for classification)
# knn = KNN(k=5)  # Example: 5 nearest neighbors
# knn.fit(latent_representation, y)

# # 10. Predict track genres using KNN
# y_pred = knn.predict(latent_representation)

# # 11. Evaluate the model (you may add your own evaluation metrics here)
# print(f"Predicted Genres: {y_pred}")


# Get the latent representation for training and testing datasets
X_train_latent = autoencoder.get_latent(X_train)
X_test_latent = autoencoder.get_latent(X_val)

# Train KNN on reduced dataset
knn_reduced = KNeighborsClassifier(n_neighbors=5)
knn_reduced.fit(X_train_latent, y_train)
y_pred_reduced = knn_reduced.predict(X_test_latent)

# Calculate metrics for the reduced dataset
accuracy_reduced = accuracy_score(y_val, y_pred_reduced)
precision_reduced = precision_score(y_val, y_pred_reduced, average='weighted')
recall_reduced = recall_score(y_val, y_pred_reduced, average='weighted')
f1_reduced = f1_score(y_val, y_pred_reduced, average='weighted')

print("\nMetrics for Reduced Dataset:")
print(f"Accuracy: {accuracy_reduced:.4f}")
print(f"Precision: {precision_reduced:.4f}")
print(f"Recall: {recall_reduced:.4f}")
print(f"F1 Score: {f1_reduced:.4f}")



# # Create a mapping from category to integer
# labels = y.unique()
# label_map = {label: idx for idx, label in enumerate(labels)}
# print("Label map:", label_map)

# y_encoded = y.map(label_map).values




# X_train,X_test,y_train, y_val = train_test_split(X,y_encoded, test_size=0.2, random_state=42)

# X_train_latent = autoencoder.get_latent(X_train)
# X_test_latent = autoencoder.get_latent(X_val)

# # Instantiate the KNN class
# knn_model = KNN(k=71, distance_metric='euclidean')

# # Fit the model on the training data
# knn_model.fit(X_train_latent, y_train)

# # Evaluate the model on the test set
# accuracy, precision, recall, f1 = knn_model.evaluate(X_test_latent, y_val)

# # Print the evaluation metrics
# print("KNN Performance on Reduced Dataset:")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")




sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/MLP')))
from MLP import MLPClassifier 


# Determine the number of unique classes
num_classes = len(np.unique(y_train))  # This will give you the number of unique classes in Y_train
print(num_classes)
# Print shapes of the training and validation sets
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)


wandb.init(project="AutoEncoder+MLP")

#Step 3: Create and fit the model using the best parameters
best_model = MLPClassifier(
    input_size=X_train.shape[1],
    output_size= num_classes,  # Output should match one-hot encoded dimension
    hidden_layers=[64,32],
    learning_rate=0.01,
    activation='tanh',
    optimizer='mini_batch',
    epochs = 500
)

# Fit the model
best_model.fit(X_train, y_train, X_val=X_val, Y_val=y_val)

# Step 4: Evaluate the model on the test set
Y_test_pred = best_model.predict(X_val)  # No need to use argmax here, predict handles it

# Calculate accuracy and other metrics
precision, recall, f1_score = best_model.calculate_metrics(y_val, Y_test_pred)

print(f"Test Accuracy: {best_model.calculate_accuracy(y_val, Y_test_pred)}")
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")



wandb.finish()