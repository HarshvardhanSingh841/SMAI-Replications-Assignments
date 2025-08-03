import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the path to the linearregression module to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/linear-regression')))

from linearregression import LinearRegression
import imageio



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  # Ensure this import matches your actual module location

def simple_regression():
    # Load the data
    data = pd.read_csv('../../data/external/linreg.csv')
    X = data.iloc[:, 0].values.reshape(-1, 1)  # Independent variable
    y = data.iloc[:, 1].values  # Dependent variable

    # Shuffle the data
    np.random.seed(42)  # For reproducibility
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split the data into train (80%), validation (10%), and test (10%)
    train_size = int(0.8 * len(X_shuffled))
    val_size = int(0.1 * len(X_shuffled))

    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]
    X_val = X_shuffled[train_size:train_size + val_size]
    y_val = y_shuffled[train_size:train_size + val_size]
    X_test = X_shuffled[train_size + val_size:]
    y_test = y_shuffled[train_size + val_size:]

    # Function to evaluate model performance
    def evaluate_and_plot(X_train, y_train, X_val, y_val, X_test, y_test, degrees, learning_rates):
        best_degree = None
        best_lr = None
        best_mse = float('inf')
        best_model = None
        
        for degree in degrees:
            for lr in learning_rates:
                model = LinearRegression(degree=degree, learning_rate=lr, num_iterations=1000)
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
                
                mse_train = model.mse(y_train, y_train_pred)
                mse_val = model.mse(y_val, y_val_pred)
                mse_test = model.mse(y_test, y_test_pred)
                
                std_train = model.std_deviation(y_train, y_train_pred)
                std_val = model.std_deviation(y_val, y_val_pred)
                std_test = model.std_deviation(y_test, y_test_pred)
                
                variance_train = model.variance(y_train, y_train_pred)
                variance_val = model.variance(y_val, y_val_pred)
                variance_test = model.variance(y_test, y_test_pred)
                
                print(f"Degree {degree}, Learning Rate {lr}:")
                print(f"Train MSE: {mse_train}, Std Dev: {std_train}, Variance: {variance_train}")
                print(f"Validation MSE: {mse_val}, Std Dev: {std_val}, Variance: {variance_val}")
                print(f"Test MSE: {mse_test}, Std Dev: {std_test}, Variance: {variance_test}")
                
                # Save the model with the best test MSE
                if mse_test < best_mse:
                    best_mse = mse_test
                    best_degree = degree
                    best_lr = lr
                    best_model = model
        
        # Save the parameters of the best model
        if best_model is not None:
            np.save('best_model_params.npy', best_model.theta_)
        
        print(f"Best Degree: {best_degree} with Learning Rate: {best_lr} and Test MSE: {best_mse}")
        
        # Plot
        plt.figure(figsize=(12, 8))
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        
        for degree in degrees:
            model = LinearRegression(degree=degree, learning_rate=best_lr, num_iterations=1000)
            model.fit(X_train, y_train)
            y_plot = model.predict(X_plot)
            plt.plot(X_plot, y_plot, label=f'Degree {degree}')
        
        plt.scatter(X_train, y_train, color='blue', label='Train Data')
        plt.scatter(X_val, y_val, color='green', label='Validation Data')
        plt.scatter(X_test, y_test, color='red', label='Test Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Polynomial Regression Fits')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Evaluate and plot for polynomial degrees from 1 to 5 and learning rates
    degrees = range(1,2)
    learning_rates = [0.001, 0.01, 0.1]
    evaluate_and_plot(X_train, y_train, X_val, y_val, X_test, y_test, degrees, learning_rates)

def polynomial_regression():
    # Load the data
    data = pd.read_csv('../../data/external/linreg.csv')
    X = data.iloc[:, 0].values.reshape(-1, 1)  # Independent variable
    y = data.iloc[:, 1].values  # Dependent variable

    # Shuffle the data
    np.random.seed(42)  # For reproducibility
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split the data into train (80%), validation (10%), and test (10%)
    train_size = int(0.8 * len(X_shuffled))
    val_size = int(0.1 * len(X_shuffled))

    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]
    X_val = X_shuffled[train_size:train_size + val_size]
    y_val = y_shuffled[train_size:train_size + val_size]
    X_test = X_shuffled[train_size + val_size:]
    y_test = y_shuffled[train_size + val_size:]

    # Function to evaluate model performance
    def evaluate_and_plot(X_train, y_train, X_val, y_val, X_test, y_test, degrees, learning_rates):
        best_degree = None
        best_lr = None
        best_mse = float('inf')
        best_model = None
        
        for degree in degrees:
            for lr in learning_rates:
                model = LinearRegression(degree=degree, learning_rate=lr, num_iterations=1000)
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
                
                mse_train = model.mse(y_train, y_train_pred)
                mse_val = model.mse(y_val, y_val_pred)
                mse_test = model.mse(y_test, y_test_pred)
                
                std_train = model.std_deviation(y_train, y_train_pred)
                std_val = model.std_deviation(y_val, y_val_pred)
                std_test = model.std_deviation(y_test, y_test_pred)
                
                variance_train = model.variance(y_train, y_train_pred)
                variance_val = model.variance(y_val, y_val_pred)
                variance_test = model.variance(y_test, y_test_pred)
                
                print(f"Degree {degree}, Learning Rate {lr}:")
                print(f"Train MSE: {mse_train}, Std Dev: {std_train}, Variance: {variance_train}")
                print(f"Validation MSE: {mse_val}, Std Dev: {std_val}, Variance: {variance_val}")
                print(f"Test MSE: {mse_test}, Std Dev: {std_test}, Variance: {variance_test}")
                
                # Save the model with the best test MSE
                if mse_test < best_mse:
                    best_mse = mse_test
                    best_degree = degree
                    best_lr = lr
                    best_model = model
        
        # Save the parameters of the best model
        if best_model is not None:
            np.save('best_model_params.npy', best_model.theta_)
        
        print(f"Best Degree: {best_degree} with Learning Rate: {best_lr} and Test MSE: {best_mse}")
        
        # Plot
        plt.figure(figsize=(12, 8))
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        
        for degree in degrees:
            model = LinearRegression(degree=degree, learning_rate=best_lr, num_iterations=1000)
            model.fit(X_train, y_train)
            y_plot = model.predict(X_plot)
            plt.plot(X_plot, y_plot, label=f'Degree {degree}')
        
        plt.scatter(X_train, y_train, color='blue', label='Train Data')
        plt.scatter(X_val, y_val, color='green', label='Validation Data')
        plt.scatter(X_test, y_test, color='red', label='Test Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Polynomial Regression Fits')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Evaluate and plot for polynomial degrees from 1 to 5 and learning rates
    degrees = range(2, 6)
    learning_rates = [0.001, 0.01, 0.1]
    evaluate_and_plot(X_train, y_train, X_val, y_val, X_test, y_test, degrees, learning_rates)



def regularization_task():
    # Load the data
    data = pd.read_csv('../../data/external/regularisation.csv')
    X = data.iloc[:, 0].values.reshape(-1, 1)  # Independent variable
    y = data.iloc[:, 1].values  # Dependent variable

    # Shuffle and split the data
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    train_size = int(0.8 * len(X_shuffled))
    val_size = int(0.1 * len(X_shuffled))
    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]
    X_val = X_shuffled[train_size:train_size + val_size]
    X_test = X_shuffled[train_size + val_size:]
    y_val = y_shuffled[train_size:train_size + val_size]
    y_test = y_shuffled[train_size + val_size:]

    # Visualize the training dataset
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Training Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()

    def evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, degrees, learning_rate, regularization_type=None):
        best_model = None
        best_mse = float('inf')
        
        for degree in degrees:
            model = LinearRegression(degree=degree, learning_rate=learning_rate, num_iterations=1000, lambda_=0.1, regularization=regularization_type)
            model.fit(X_train, y_train)
            
            # Evaluate on validation and test sets
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            mse_train = model.mse(y_train, y_train_pred)
            mse_val = model.mse(y_val, y_val_pred)
            mse_test = model.mse(y_test, y_test_pred)
            
            std_train = model.std_deviation(y_train, y_train_pred)
            std_val = model.std_deviation(y_val, y_val_pred)
            std_test = model.std_deviation(y_test, y_test_pred)
            
            variance_train = model.variance(y_train, y_train_pred)
            variance_val = model.variance(y_val, y_val_pred)
            variance_test = model.variance(y_test, y_test_pred)
            
            print(f"Degree {degree}, Regularization {regularization_type}:")
            print(f"Train MSE: {mse_train}, Std Dev: {std_train}, Variance: {variance_train}")
            print(f"Validation MSE: {mse_val}, Std Dev: {std_val}, Variance: {variance_val}")
            print(f"Test MSE: {mse_test}, Std Dev: {std_test}, Variance: {variance_test}")
            
            # Track the best model based on validation MSE
            if mse_val < best_mse:
                best_mse = mse_val
                best_model = model

        return best_model

    def plot_fits(X_train, y_train, degrees, learning_rate, regularization_type=None):
        plt.figure(figsize=(12, 8))
        X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
        
        for degree in degrees:
            model = LinearRegression(degree=degree, learning_rate=learning_rate, num_iterations=1000, lambda_=0.1, regularization=regularization_type)
            model.fit(X_train, y_train)
            y_plot = model.predict(X_plot)
            plt.plot(X_plot, y_plot, label=f'Degree {degree} ({regularization_type})')
        
        plt.scatter(X_train, y_train, color='blue', label='Training Data')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Polynomial Fits with {regularization_type if regularization_type else "No Regularization"}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Define degrees to evaluate
    degrees = range(1, 21)
    learning_rate = 0.01

    # Evaluate and plot fits for different types of regularization
    print("Evaluating models with no regularization:")
    best_model_no_reg = evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, degrees, learning_rate, regularization_type=None)
    plot_fits(X_train, y_train, degrees, learning_rate, regularization_type=None)
    
    print("Evaluating models with L1 regularization:")
    best_model_l1 = evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, degrees, learning_rate, regularization_type='l1')
    plot_fits(X_train, y_train, degrees, learning_rate, regularization_type='l1')
    
    print("Evaluating models with L2 regularization:")
    best_model_l2 = evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, degrees, learning_rate, regularization_type='l2')
    plot_fits(X_train, y_train, degrees, learning_rate, regularization_type='l2')






if __name__ == "__main__":
    #task 3.1.1
    #
    simple_regression()
    #task 3.1.2
    #polynomial_regression()
    #task 3.2.1
    #regularization_task()

