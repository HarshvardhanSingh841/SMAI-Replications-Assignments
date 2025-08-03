import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/MLP')))
from MLP import MLPClassifier 

# Load the dataset
data_path = os.path.join('..', '..', 'data', 'external', 'WineQT.csv')
df = pd.read_csv(data_path)

# Get the descriptive statistics for all attributes
description = df.describe()
pd.set_option('display.max_columns', None)
# Show the mean, std, min, and max

# Print only the 'mean', 'std', 'min', and 'max' rows
print(description.loc[['mean', 'std', 'min', 'max']])

# Drop the 'Id' column first
plot = df.drop(columns='Id')

# Get the descriptive statistics
description = plot.describe()

# Select the 'mean', 'std', 'min', and 'max' rows
stats_to_plot = description.loc[['mean', 'std', 'min', 'max']]

# Plotting the statistics
plt.figure(figsize=(12, 8))

# Transpose the stats for better plotting (so columns become x-axis labels)
stats_to_plot.T.plot(kind='bar', figsize=(12, 8), width=0.8)

plt.title('Mean, Standard Deviation, Min, and Max Values for Wine Attributes')
plt.xlabel('Wine Attributes')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()


plt.figure(figsize=(8, 6))
df['quality'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of Wine Quality Labels')
plt.xlabel('Wine Quality')
plt.ylabel('Count')
plt.show()



# Load dataset and drop 'Id'
df = df.drop(columns='Id')
# Separate features (excluding 'quality' column)
features = df.drop(columns='quality')

# Step 1: Normalize the data (scale to [0, 1] range)
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(features)

# Step 2: Standardize the normalized data (zero mean, unit variance)
scaler = StandardScaler()
normalized_then_standardized_data = scaler.fit_transform(normalized_data)

# Convert back to a DataFrame
final_df = pd.DataFrame(normalized_then_standardized_data, columns=features.columns)

# Add the 'quality' column back to the DataFrame
final_df['quality'] = df['quality'].values


# # One-hot encode the labels (1-8 scale) while keeping it order-independent
# # Y_encoded = pd.get_dummies(Y, drop_first=False).values  # Remove this line since you're doing it in MLP

Y = df['quality'].values  # Extract the quality column as a 1D array
# Prepare your dataset (X_train, Y_train, X_val, Y_val)
X_train, X_val, Y_train, Y_val = train_test_split(normalized_then_standardized_data, Y, test_size=0.2, random_state=42)

# Determine the number of unique classes
num_classes = len(np.unique(Y_train))  # This will give you the number of unique classes in Y_train

# Print shapes of the training and validation sets
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)





# import wandb

# # Step 1: Define the sweep configuration in code
# sweep_config = {
#     'method': 'bayes',
#     'metric': {
#         'goal': 'maximize',
#         'name': 'Validation Accuracy',
#     },
#     'parameters': {
#         'learning_rate': {
#           'values':  [0.001,0.01,0.1],
#         },
#         'hidden_layers': {
#             'values': [[9], [18, 9], [64, 32]],
#         },
#         'activation_function': {
#             'values': ['relu', 'sigmoid', 'tanh'],
#         },
#         'optimizer': {
#             'values': ['sgd', 'mini_batch', 'batch'],
#         },
#     }
# }

# # Step 2: Initialize the sweep
# sweep_id = wandb.sweep(sweep_config, project="mlp-classification")

# # Step 3: Define the function to run for each sweep
# def train():
#     # Initialize W&B
#     wandb.init()

#     # Access hyperparameters from the sweep config
#     config = wandb.config

#     # Instantiate your model with hyperparameters from the config
#     model = MLPClassifier(input_size=X_train.shape[1],
#                           output_size=num_classes,
#                           hidden_layers=config.hidden_layers,
#                           learning_rate=config.learning_rate,
#                           activation=config.activation_function,
#                           optimizer=config.optimizer,
#                           epochs=500)

#     # Fit the model
#     model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val)

#     # Log metrics
#     train_accuracy = model.calculate_accuracy(Y_train, model.forward_propagation(X_train)[0][-1])
#     val_accuracy = model.calculate_accuracy(Y_val, model.forward_propagation(X_val)[0][-1])
    
#     wandb.log({
#         'Train Accuracy': train_accuracy,
#         'Validation Accuracy': val_accuracy,
#     })

#     wandb.finish()  # End W&B logging

# # Step 4: Run the agent to start the sweep
# wandb.agent(sweep_id, function=train, count=30)  # count specifies how many runs to execute


X_train_val , X_test , Y_train_val , Y_test = train_test_split(normalized_then_standardized_data, Y, test_size=0.1, random_state=42)

X_train , X_val , Y_train , Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1, random_state=42)

import wandb
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Assuming you have already loaded and preprocessed your data
# X_processed: Your features, Y: Your labels (one-hot encoded)


# Initialize W&B before starting training
wandb.init(project='MLPclassifier Analysis')  # Adjust the project and run name as needed
# Step 3: Create and fit the model using the best parameters
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
best_model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val)

# Step 4: Evaluate the model on the test set
Y_test_pred = best_model.predict(X_test)  # No need to use argmax here, predict handles it

# Calculate accuracy and other metrics
precision, recall, f1_score = best_model.calculate_metrics(Y_test, Y_test_pred)

print(f"Test Accuracy: {best_model.calculate_accuracy(Y_test, Y_test_pred)}")
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

# Step 1: Predict the classes for all data points
#predictions = MLPClassifier.predict(X_test)

# Step 2: Convert predictions back from one-hot encoding if needed
predicted_classes = Y_test_pred   # +1 if your classes are from 1 to 8
true_classes = (Y_test) 

# Step 3: Calculate and print classification metrics
print(classification_report(true_classes, predicted_classes))

# Step 4: Visualize the results with a confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(true_classes))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Step 5: Identify classes with high misclassification
# Calculate per-class accuracy
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("Per-Class Accuracy:", per_class_accuracy)

# Identify classes with low accuracy
low_accuracy_classes = np.where(per_class_accuracy < 0.5)[0]  # Threshold of 0.5
print("Classes with low accuracy:", low_accuracy_classes)

wandb.finish()


# activations = ['relu', 'sigmoid', 'tanh', 'linear']
# for activation in activations:
#     run = wandb.init(project="hyperparameter_analysis", name=f"Activation-{activation}", reinit=True)
#     model = MLPClassifier(input_size=X_train.shape[1], output_size=num_classes, hidden_layers=[64, 32], 
#                           learning_rate=0.01, activation=activation, optimizer='mini_batch', batch_size=32, epochs=500)
#     model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val)
#     run.finish()


# learning_rates = [0.001, 0.01, 0.1, 0.5]
# for lr in learning_rates:
#     run = wandb.init(project="hyperparameter_analysis", name=f"LearningRate-{lr}", reinit=True)
#     model = MLPClassifier(input_size=X_train.shape[1], output_size=num_classes, hidden_layers=[64, 32], 
#                           learning_rate=lr, activation='tanh', optimizer='mini_batch', batch_size=32, epochs=500)
#     model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val)
#     run.finish()

# batch_sizes = [16, 32, 64, 128]
# for batch_size in batch_sizes:
#     run = wandb.init(project="hyperparameter_analysis", name=f"BatchSize-{batch_size}", reinit=True)
#     model = MLPClassifier(input_size=X_train.shape[1], output_size=num_classes, hidden_layers=[64, 32], 
#                           learning_rate=0.01, activation='tanh', optimizer='mini_batch', batch_size=batch_size, epochs=500)
#     model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val)
#     run.finish()



# from MLMLP import MLPMultiLabelClassifier
# from sklearn.preprocessing import MultiLabelBinarizer


# data_path3 = os.path.join('..', '..', 'data', 'external', 'advertisement.csv')
# df= pd.read_csv(data_path3)

# # Step 2: Identify categorical columns
# categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# # Step 3: Create custom mappings for non-target categorical columns
# custom_mappings = {}
# for col in categorical_cols:
#     unique_values = np.unique(df[col])
#     # Create a mapping for each unique value to a unique integer
#     custom_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}

# # Step 4: Replace categorical values with their corresponding numerical representations
# for col, mapping in custom_mappings.items():
#     df[col] = df[col].replace(mapping)

# # Step 5: Process multi-label target column (assuming it's named 'labels')
# # First, ensure all values are strings, handle missing values as needed
# df['labels'] = df['labels'].astype(str)  # Convert all values to strings
# df['labels'] = df['labels'].replace('nan', '')  # Replace 'nan' strings with empty strings

# # Now split labels into lists while filtering out empty strings
# df['labels'] = df['labels'].apply(lambda x: x.split(',') if x else [])

# # Use MultiLabelBinarizer to encode the labels
# mlb = MultiLabelBinarizer()
# Y = mlb.fit_transform(df['labels'])

# # Verify the result
# print("Encoded labels shape:", Y.shape)
# print("Unique classes:", mlb.classes_)
# # Step 6: Drop the label column from features
# X = df.drop(columns=['labels'])

# # Convert DataFrame to NumPy array if necessary
# X_processed = X.to_numpy()

# # X_processed is now a numerical matrix ready for your MLP
# print("Processed feature matrix shape:", X_processed.shape)
# print("Binary encoded labels shape:", Y.shape)

# # Example of using the processed features and labels with your MLP
# # Initialize your MLP classifier
# mlp_classifier = MLPMultiLabelClassifier(input_size=X_processed.shape[1], output_size=len(mlb.classes_))

# # Train your MLP model with the binary labels
# mlp_classifier.fit(X_processed, Y, X_val=None, Y_val=None, early_stopping=True)

# # Optionally, save your trained model, log metrics, etc.