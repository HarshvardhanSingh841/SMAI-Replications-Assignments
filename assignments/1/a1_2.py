import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# Add the path to the knn module to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/knn')))
from knn import KNN



# Function to load the dataset
def load_data():
    data_path = os.path.join('..', '..', 'data', 'external', 'spotify.csv')
    df = pd.read_csv(data_path)
    return df

# Function to perform EDA (Exploratory Data Analysis)
def perform_eda(df):
    # Display the first few rows to understand the data structure
    print("First few rows of the dataset:")
    print(df.head())

    # Display basic statistics of the dataset
    print("\nSummary statistics of the dataset:")
    print(df.describe(include='all'))  # Include all data types

    # Display the number of missing values in each column
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Display all column names
    print("\nColumn names:")
    print(df.columns)

    # Set up the style for seaborn
    sns.set(style="whitegrid")

    # List of numerical columns
    numerical_columns = [
        'popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'time_signature'
    ]

    # Check if 'track_genre' column exists
    if 'track_genre' not in df.columns:
        print("\nError: 'track_genre' column not found in the DataFrame.")
        return

    # Normalize numerical columns using vectorized operations
    min_vals = df[numerical_columns].min()
    max_vals = df[numerical_columns].max()
    normalized_df = df.copy()
    normalized_df[numerical_columns] = (df[numerical_columns] - min_vals) / (max_vals - min_vals)

    # Manually encode the 'track_genre' column
    genre_mapping = {genre: idx for idx, genre in enumerate(df['track_genre'].unique())}
    normalized_df['track_genre_encoded'] = df['track_genre'].map(genre_mapping)

    # Display histograms for normalized features
    plt.figure(figsize=(20, 15))
    normalized_df[numerical_columns].hist(bins=20, figsize=(20, 15))
    plt.suptitle("Feature Distributions (Normalized Data)")
    plt.savefig(os.path.join('figures', 'histograms_normalized.png'))
    plt.show()

    # Display boxplots for normalized numerical features
    plt.figure(figsize=(20, 15))
    sns.boxplot(data=normalized_df[numerical_columns])
    plt.title("Boxplot of Normalized Numerical Features")
    plt.savefig(os.path.join('figures', 'boxplots_normalized.png'))
    plt.show()

    # Compute correlation matrix using normalized columns
    numeric_df = normalized_df[numerical_columns + ['track_genre_encoded']]
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join('figures', 'correlation_matrix.png'))
    plt.show()

    # Generate the pair plot for all numerical features using the full dataset
    # Ensure all selected features are in the DataFrame
    if all(feature in df.columns for feature in numerical_columns + ['track_genre_encoded']):
        # Generate the pair plot with full dataset
        sns.pairplot(normalized_df[numerical_columns + ['track_genre_encoded']], hue='track_genre_encoded', palette='tab10', diag_kind='kde')
        plt.savefig(os.path.join('figures', 'pairplot.png'))
        plt.show()
    else:
        print("\nSome selected features are not in the DataFrame columns.")

    # Identify features with high correlation to the target variable
    if 'track_genre_encoded' in numeric_df.columns:
        target_corr = corr_matrix['track_genre_encoded'].abs().sort_values(ascending=False)
        print("\nFeatures most correlated with the target variable (track_genre_encoded):")
        print(target_corr)
    else:
        print("\nTarget variable 'track_genre_encoded' is not numeric or does not exist.")


def perform_hyperparameter_tuning(df):
    # Print column names for verification
    print("Columns in DataFrame:", df.columns)

    # Ensure 'track_genre' exists in the DataFrame
    if 'track_genre' not in df.columns:
        print("Error: 'track_genre' column not found in the DataFrame.")
        return

    # Convert 'explicit' column to numeric
    df['explicit'] = df['explicit'].astype(int)

    # Encode categorical target variable
    df_encoded = pd.get_dummies(df, columns=['track_genre'], drop_first=True)

    # Drop non-numeric columns
    non_numeric_cols = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name']
    df_numeric = df_encoded.drop(columns=non_numeric_cols)

    # Convert all columns to numeric, if they are not already
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # Check for columns with boolean data types
    bool_cols = df_numeric.select_dtypes(include='bool').columns
    for col in bool_cols:
        df_numeric[col] = df_numeric[col].astype(int)  # Convert booleans to integers

    # Normalize numeric columns
    numeric_columns = df_numeric.columns
    for col in numeric_columns:
        if df_numeric[col].dtype in [np.float64, np.int64]:  # Ensure column is numeric
            min_val = df_numeric[col].min()
            max_val = df_numeric[col].max()
            # Avoid division by zero if max_val equals min_val
            if max_val != min_val:
                df_numeric[col] = (df_numeric[col] - min_val) / (max_val - min_val)
            else:
                df_numeric[col] = 0  # Set to 0 or another constant if no range

    # Check column names after encoding and normalization
    print("Columns after encoding and normalization:", df_numeric.columns)

    # Extract feature columns and target column
    feature_columns = [col for col in df_numeric.columns if not col.startswith('track_genre')]
    X = df_numeric[feature_columns].values
    y = df_numeric[[col for col in df_numeric.columns if col.startswith('track_genre')]].values

    # Ensure X and y are of numeric types
    print("X data type:", X.dtype)
    print("y data type:", y.dtype)

    # 80:10:10 train:test:validation split
    def train_test_split_custom(X, y, test_size=0.2, val_size=0.5, random_state=None):
        if random_state:
            np.random.seed(random_state)
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        test_split = int(num_samples * test_size)
        val_split = int(test_split * val_size)
        X_train = X[:-test_split]
        y_train = y[:-test_split]
        X_test = X[-test_split:]
        y_test = y[-test_split:]
        X_val = X_test[:val_split]
        y_val = y_test[:val_split]
        X_test = X_test[val_split:]
        y_test = y_test[val_split:]
        return X_train, X_test, X_val, y_train, y_test, y_val

    X_train, X_test, X_val, y_train, y_test, y_val = train_test_split_custom(X, y, random_state=42)

    k_values = [39,49,71,109,101,43,57,33,51,11]  # Only odd k values
    distance_metrics = ['euclidean','manhattan']
    results = []

    for distance_metric in distance_metrics:
        for k in k_values:
            model = KNN(k=k, distance_metric=distance_metric)
            print(f"{k}")
            model.fit(X_train, y_train)
            accuracy, precision, recall, f1 = model.evaluate(X_val, y_val)
            results.append((k, distance_metric, accuracy, precision, recall, f1))

    # Find top 10 combinations
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    top_10 = sorted_results[:10]

    # Print top 10 combinations
    print("Top 10 {k, distance_metric} pairs:")
    for result in top_10:
        print(f"k={result[0]}, distance_metric={result[1]}, Accuracy={result[2]:.4f}, Precision={result[3]:.4f}, Recall={result[4]:.4f}, F1 Score={result[5]:.4f}")

    # Plot k vs accuracy for a chosen distance metric
    chosen_metric = 'euclidean'  # Choose distance metric for plotting
    k_values_plot = [result[0] for result in results if result[1] == chosen_metric]
    accuracies = [result[2] for result in results if result[1] == chosen_metric]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values_plot, accuracies, marker='o')
    plt.title(f'k vs Accuracy ({chosen_metric} Distance Metric)')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()



# Driver code
if __name__ == "__main__":
   #task 1 (2.2.1) 
    # Load the dataset
    #df = load_data()
    # Perform Exploratory Data Analysis
    #perform_eda(df)
   #task 2 (2.2.4)
    df = pd.read_csv('../../data/external/spotify.csv')  # Adjust path if necessary
    #Perform Hyperparameter Tuning
    perform_hyperparameter_tuning(df)

