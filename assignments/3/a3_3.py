import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/MLP')))
from MLPR import MLPRegressor

# Load the dataset
data_path = os.path.join('..', '..', 'data', 'external', 'HousingData.csv')
df = pd.read_csv(data_path)

# Replace 'Na' with NaN (if 'Na' is not recognized as NaN by pandas automatically)
df.replace('NA', pd.NA, inplace=True)

# Option 1: Replace missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

# Option 2: Replace missing values with the median of each column
# df.fillna(df.median(), inplace=True)

# Check if there are any remaining missing values
print(df.isna().sum())
print(df.head)


# Describe the dataset
description = df.describe().T[['mean', 'std', 'min', 'max']]
print(description)

# Plot distribution of MEDV
plt.figure(figsize=(8, 6))
plt.hist(df['MEDV'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of MEDV (Median Value of Owner-Occupied Homes)')
plt.xlabel('MEDV ($1000s)')
plt.ylabel('Frequency')
plt.show()



# Features and target
X = df.drop(columns=['MEDV'])  # Drop the target column
y = df['MEDV'].values  # Target variable


# Step 1: Normalize the data (scale to [0, 1] range)
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(X)

# Step 2: Standardize the normalized data (zero mean, unit variance)
scaler = StandardScaler()
X = scaler.fit_transform(normalized_data)

# Check scaled data
print("Mean after scaling (train):", X.mean(axis=0))
print("Standard deviation after scaling (train):", X.std(axis=0))


#80:10:10 split for train test val

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

y_train = y_train.reshape(-1, 1)  # Shape: (404, 1)
y_val = y_val.reshape(-1, 1)      # Shape: (51, 1)
y_test = y_test.reshape(-1,1) 
print("Training set size:", X_train.shape)
print(y_val.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)

# wandb.init(project='MLPRegressor')
# mlp = MLPRegressor(input_size=X_train.shape[1], hidden_layers=[64, 32], output_size=1, epochs = 1000,
#                    activation='tanh', learning_rate=0.0390845, optimizer='mini_batch', batch_size=32)
# mlp.fit(X_train, y_train, X_val=X_val, Y_val=y_val ,early_stopping = True , patience = 50)


# wandb.finish()




# # Step 1: Define the sweep configuration for the MLP Regressor
# sweep_config = {
#     'method': 'bayes',  # Bayesian optimization for efficient hyperparameter tuning
#     'metric': {
#         'goal': 'minimize',  # We aim to minimize the validation MSE
#         'name': 'Validation MSE',
#     },
#     'parameters': {
#         'learning_rate': {
#             'values': [0.001, 0.01, 0.1],  # Testing different learning rates
#         },
#         'hidden_layers': {
#             'values': [[9], [18, 9], [64, 32]],  # Different architectures for hidden layers
#         },
#         'activation_function': {
#             'values': ['relu', 'sigmoid', 'tanh'],  # Activation functions
#         },
#         'optimizer': {
#             'values': ['sgd', 'mini_batch', 'batch'],  # Optimizers
#         },
#     }
# }

# # Step 2: Initialize the sweep
# sweep_id = wandb.sweep(sweep_config, project="mlp-regression")

# # Step 3: Define the training function
# def train():
#     # Initialize W&B
#     wandb.init()
    
#     # Access the hyperparameters from the sweep config
#     config = wandb.config
    
#     # Instantiate your MLPRegressor model with hyperparameters from the sweep config
#     model = MLPRegressor(input_size=X_train.shape[1],
#                          output_size=1,  # Regression output size is 1 (for the price prediction)
#                          hidden_layers=config.hidden_layers,
#                          learning_rate=config.learning_rate,
#                          activation=config.activation_function,
#                          optimizer=config.optimizer,
#                          epochs=1000)  # Define your chosen number of epochs
    
#     # Train the model
#     model.fit(X_train, y_train, X_val=X_val, Y_val=y_val)
    
#     # Evaluate model performance on training and validation sets
#     y_train_pred = model.forward_propagation(X_train)[0][-1]
#     y_val_pred = model.forward_propagation(X_val)[0][-1]
    
#     # Calculate MSE, RMSE, and R-squared for both training and validation sets
#     train_mse = np.mean((y_train - y_train_pred) ** 2)
#     val_mse = np.mean((y_val - y_val_pred) ** 2)
    
#     train_rmse = np.sqrt(train_mse)
#     val_rmse = np.sqrt(val_mse)
    
#     train_r2 = 1 - np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
#     val_r2 = 1 - np.sum((y_val - y_val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
    
#     # Log metrics to W&B
#     wandb.log({
#         'Train MSE': train_mse,
#         'Validation MSE': val_mse,
#         'Train RMSE': train_rmse,
#         'Validation RMSE': val_rmse,
#         'Train R-squared': train_r2,
#         'Validation R-squared': val_r2,
#     })
    
#     wandb.finish()  # End W&B logging

# # Step 4: Run the W&B sweep agent
# wandb.agent(sweep_id, function=train, count=30)  # count specifies the number of sweep runs



#Best model


# wandb.init(project="best model")
# best_model = MLPRegressor(input_size=X_train.shape[1],
#                      output_size=1,  # Regression output size is 1 (for the price prediction)
#                         hidden_layers=[18,9],
#                         learning_rate=0.01,
#                         activation='tanh',
#                         optimizer='mini_batch',
#                         epochs=1000)  # Define your chosen number of epochs
# best_model.fit(X_train,y_train,X_val = X_val ,Y_val = y_val,early_stopping = True , patience = 50)

# Y_pred_test = best_model.predict(X_test)

# # Calculate MSE, RMSE, and MAE
# mse_test = np.mean((y_test - Y_pred_test) ** 2)
# rmse_test = np.sqrt(mse_test)
# mae_test = np.mean(np.abs(y_test - Y_pred_test))

# print(f"Test MSE: {mse_test}")
# print(f"Test RMSE: {rmse_test}")
# print(f"Test MAE: {mae_test}")

# wandb.finish()


#logistic regression

data_path2 = os.path.join('..', '..', 'data', 'external', 'diabetes.csv')
df2 = pd.read_csv(data_path2)

# Describe the dataset
description = df2.describe().T[['mean', 'std', 'min', 'max']]
print(description)

# Plot distribution of MEDV
plt.figure(figsize=(8, 6))
plt.hist(df2['Outcome'], bins=2, color='skyblue', edgecolor='black')
plt.title('outcome of diabetes')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.show()



X = df2.drop(columns=['Outcome'])  # Drop the target column
y = df2['Outcome'].values  # Target variable


# Step 1: Normalize the data (scale to [0, 1] range)
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(X)

# Step 2: Standardize the normalized data (zero mean, unit variance)
scaler = StandardScaler()
X = scaler.fit_transform(normalized_data)

# Check scaled data
print("Mean after scaling (train):", X.mean(axis=0))
print("Standard deviation after scaling (train):", X.std(axis=0))


#80:10:10 split for train test val

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

y_train = y_train.reshape(-1, 1)  # Shape: (404, 1)
y_val = y_val.reshape(-1, 1)      # Shape: (51, 1)
y_test = y_test.reshape(-1,1) 
print("Training set size:", X_train.shape)
print(y_val.shape)
print("Validation set size:", X_val.shape)
print("Test set size:", X_test.shape)



wandb.init(project="MLP Logistic")
from MLPR import MLPLogistic
# Create an instance of your MLP for logistic regression
model_bce = MLPLogistic(
    input_size=X_train.shape[1],  # Number of features in the input
    output_size=1,  # Single output for binary classification
    hidden_layers=[],  # No hidden layers
    learning_rate=0.001,
    activation='sigmoid',  # Sigmoid for binary output
    optimizer='sgd',  # or any optimizer you prefer
    epochs=500
)

# Fit the model using BCE loss
model_bce.fit(X_train, y_train,X_val=X_val,Y_val = y_val, loss_function='bce')



wandb.finish()




import numpy as np

# Assuming you have X_test and y_test (the true labels)

# Get predictions from your model
predictions = model_bce.predict(X_test)
predictions = predictions.reshape(-1,1)

# Calculate MSE for each data point
bce_losses = -(y_test * np.log(predictions + 1e-15) + (1 - y_test) * np.log(1 - predictions + 1e-15))/predictions.shape[0]
  # Convert to 1D array for easier analysis



print("y_test shape:", y_test.shape)
print("predictions shape:", predictions.shape)
print("mse_losses shape:", bce_losses.shape)
# Create a DataFrame for better visualization
import pandas as pd

results_df = pd.DataFrame({'True Values': y_test.flatten(), 'Predictions': predictions.flatten(), 'BCE Loss': bce_losses.flatten()})

import matplotlib.pyplot as plt

plt.scatter(results_df['True Values'], results_df['BCE Loss'], alpha=0.5)
plt.title('BCE Loss vs True Values')
plt.xlabel('True Values')
plt.ylabel('BCE Loss')
plt.grid()
plt.show()