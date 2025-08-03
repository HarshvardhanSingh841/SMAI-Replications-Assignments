import numpy as np
import wandb

class MLPClassifier:
    def __init__(self, input_size, output_size, hidden_layers=[64, 64], learning_rate=0.01, 
                 activation='relu', optimizer='sgd', batch_size=32, epochs=100):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialize_parameters()

        self.class_labels = None

    def initialize_parameters(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i+1])))

    def forward_propagation(self, X):
        activations = [X]  # List to store activations layer by layer
        Zs = []  # List to store linear transformations Z = W.X + b

        for i in range(len(self.weights)):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            Zs.append(Z)
        
            if i == len(self.weights) - 1:
                A = self.softmax(Z)  # Apply softmax to the output layer
            else:
                A = self.activate(Z, self.activation)  # Hidden layers use the selected activation function
            activations.append(A)

        return activations, Zs
    
    def activate(self, Z, activation_function):
        if activation_function == 'relu':
            return np.maximum(0, Z)
        elif activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation_function == 'tanh':
            return np.tanh(Z)
        elif activation_function == 'linear':
            return Z

    def backward_propagation(self, X, Y, activations, Zs):
        m = X.shape[0]  
        dW = []
        db = []
        dA = activations[-1] - Y  # Loss derivative with respect to final activation

        for i in reversed(range(len(self.weights))):
            dZ = dA * self.activation_derivative(activations[i+1], self.activation)
            dW_i = np.dot(activations[i].T, dZ) / m
            db_i = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.weights[i].T)
        
            dW.insert(0, dW_i)
            db.insert(0, db_i)

        return dW, db

    def activation_derivative(self, A, activation_function):
        if activation_function == 'relu':
            return np.where(A > 0, 1, 0)
        elif activation_function == 'sigmoid':
            return A * (1 - A)
        elif activation_function == 'tanh':
            return 1 - A ** 2
        elif activation_function == 'linear':
            return 1

    # def fit(self, X, Y, X_val=None, Y_val=None, early_stopping=False, patience=5):
    #     # One-hot encode Y for training
    #     Y_encoded = self.one_hot_encode(Y, self.output_size)
    #     Y_val_encoded = self.one_hot_encode(Y_val, self.output_size) if Y_val is not None else None
        
    #     best_loss = float('inf')
    #     patience_counter = 0

    #     for epoch in range(self.epochs):
    #         activations, Zs = self.forward_propagation(X)
    #         dW, db = self.backward_propagation(X, Y_encoded, activations, Zs)

    #         self.update_parameters(dW, db)

    #         train_loss = self.calculate_loss(Y_encoded, activations[-1])
    #         wandb.log({'Epoch': epoch + 1, 'Train Loss': train_loss})

    #         train_accuracy = self.calculate_accuracy(Y, activations[-1])  # Decode predictions
    #         wandb.log({'Train Accuracy': train_accuracy})

    #         if X_val is not None and Y_val is not None:
    #             val_activations, _ = self.forward_propagation(X_val)
    #             val_loss = self.calculate_loss(Y_val_encoded, val_activations[-1])
    #             val_accuracy = self.calculate_accuracy(Y_val, val_activations[-1])
    #             wandb.log({'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy})

    #             # Metrics: precision, recall, F1 score
    #             val_predictions = self.decode_predictions(val_activations[-1])
    #             precision, recall, f1_score = self.calculate_metrics(Y_val, val_predictions)
    #             wandb.log({'Validation Precision': precision, 'Validation Recall': recall, 'Validation F1 Score': f1_score})

    #         if early_stopping:
    #             if val_loss < best_loss:
    #                 best_loss = val_loss
    #                 patience_counter = 0
    #             else:
    #                 patience_counter += 1
            
    #             if patience_counter >= patience:
    #                 print(f"Early stopping at epoch {epoch + 1}")
    #                 break

    def fit(self, X, Y, X_val=None, Y_val=None, early_stopping=False, patience=5):
        # One-hot encode Y for training
        Y_encoded = self.one_hot_encode(Y,self.output_size)
        Y_val_encoded = self.one_hot_encode(Y_val,self.output_size) if Y_val is not None else None

        best_loss = float('inf')
        patience_counter = 0

        # For batch or mini-batch gradient descent
        if self.optimizer in ['batch', 'mini_batch']:
            # Calculate number of batches
            num_batches = X.shape[0] // self.batch_size + (1 if X.shape[0] % self.batch_size != 0 else 0)

        for epoch in range(self.epochs):
            if self.optimizer == 'sgd':
                for i in range(X.shape[0]):  # Loop through each training example
                    # Forward and backward propagation for each example
                    activations, Zs = self.forward_propagation(X[i:i+1])  # Single example
                    dW, db = self.backward_propagation(X[i:i+1], Y_encoded[i:i+1], activations, Zs)
                    self.update_parameters(dW, db)  # Update parameters for each example

            elif self.optimizer in ['mini_batch', 'batch']:
                # Forward propagation for the entire batch
                activations, Zs = self.forward_propagation(X)
                dW, db = self.backward_propagation(X, Y_encoded, activations, Zs)

                if self.optimizer == 'mini_batch':
                    for batch_index in range(num_batches):
                        # Select mini-batch
                        start = batch_index * self.batch_size
                        end = start + self.batch_size
                        X_batch = X[start:end]
                        Y_batch = Y_encoded[start:end]
                    
                        # Forward and backward propagation for the mini-batch
                        activations_batch, Zs_batch = self.forward_propagation(X_batch)
                        dW_batch, db_batch = self.backward_propagation(X_batch, Y_batch, activations_batch, Zs_batch)

                        # Update parameters for mini-batch
                        self.update_parameters(dW_batch, db_batch)

                elif self.optimizer == 'batch':
                    # Update parameters for the entire batch
                    self.update_parameters(dW, db)

            # Logging and validation checks
            train_loss = self.calculate_loss(Y_encoded, activations[-1])
            wandb.log({'Epoch': epoch + 1, 'Train Loss': train_loss})

            train_accuracy = self.calculate_accuracy(Y, activations[-1])
            wandb.log({'Train Accuracy': train_accuracy})

            if X_val is not None and Y_val is not None:
                val_activations, _ = self.forward_propagation(X_val)
                val_loss = self.calculate_loss(Y_val_encoded, val_activations[-1])
                val_accuracy = self.calculate_accuracy(Y_val, val_activations[-1])
                wandb.log({'Validation Loss': val_loss, 'Validation Accuracy': val_accuracy})

                if early_stopping:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break


    def update_parameters(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def calculate_loss(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        Y_pred_clipped = np.clip(Y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / m
        return loss

    def calculate_accuracy(self, Y_true, Y_pred):
        Y_pred_decoded = self.decode_predictions(Y_pred)
        accuracy = np.sum(Y_pred_decoded == Y_true) / Y_true.shape[0]
        return accuracy

    def calculate_metrics(self, Y_true, Y_pred):
        unique_classes = np.unique(Y_true)  # Get the unique class labels
        precision_per_class = []
        recall_per_class = []
        f1_score_per_class = []

        for cls in unique_classes:
            tp = np.sum((Y_true == cls) & (Y_pred == cls))  # True Positives for this class
            fp = np.sum((Y_true != cls) & (Y_pred == cls))  # False Positives for this class
            fn = np.sum((Y_true == cls) & (Y_pred != cls))  # False Negatives for this class

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_score_per_class.append(f1_score)

        # Compute macro averages
        macro_precision = np.mean(precision_per_class)
        macro_recall = np.mean(recall_per_class)
        macro_f1_score = np.mean(f1_score_per_class)

        return macro_precision, macro_recall, macro_f1_score

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Stability improvement
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def one_hot_encode(self, Y, num_classes):
        """
        One-hot encode the target labels.
        Adjusted for target values between 3 and 8.
        """
        one_hot = np.zeros((Y.shape[0], num_classes))
        one_hot[np.arange(Y.shape[0]), Y - 3] = 1  # Subtract 3 to start indexing from 0
        return one_hot


    def one_hot_encode1(self, Y):
        """
        One-hot encode the target labels for string types.
        """
        if self.class_labels is None:
            self.class_labels = np.unique(Y)  # Store unique classes found
        class_to_index = {cls: idx for idx, cls in enumerate(self.class_labels)}
        one_hot = np.zeros((Y.shape[0], len(self.class_labels)))
        for i, label in enumerate(Y):
            one_hot[i, class_to_index[label]] = 1
        return one_hot

    def decode_predictions(self, Y_pred):
        """
        Decode predictions back to original class labels (3 to 8).
        """

        # Check if Y_pred is 1D (already decoded)
        if Y_pred.ndim == 1:
            return Y_pred  # Already decoded
        return np.argmax(Y_pred, axis=1) + 3  # Add 3 to return to the original label range (3-8)
    
    def decode_predictions1(self, Y_pred):
        if Y_pred.ndim == 1:
            return Y_pred  # Already decoded
        return self.class_labels[np.argmax(Y_pred, axis=1)]  # Return the original label

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        predictions = self.decode_predictions(activations[-1])
        return predictions
    


    def gradient_check(self, X, Y, epsilon=1e-7):
        """
        Perform gradient checking to compare the analytical and numerical gradients.
    
        Args:
            X: Input data (features).
            Y: One-hot encoded target labels.
            epsilon: Small value to compute numerical gradients.
        """
        # Convert Y to one-hot encoding if it's not already in that form
        if len(Y.shape) == 1:  # Y is not one-hot encoded yet
            Y = self.one_hot_encode1(Y)
    
        # Perform forward propagation to get activations and Z values
        activations, Zs = self.forward_propagation(X)
    
        # Compute the analytical gradients
        dW, db = self.backward_propagation(X, Y, activations, Zs)
    
        # Initialize lists for storing numerical gradients
        numerical_dW, numerical_db = [], []
    
        # Compute numerical gradients for weights
        for l in range(len(self.weights)):
            numerical_dW_l = np.zeros_like(self.weights[l])
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    # Perturb weight by epsilon (positive and negative)
                    self.weights[l][i, j] += epsilon
                    plus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                    self.weights[l][i, j] -= 2 * epsilon
                    minus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                    self.weights[l][i, j] += epsilon  # Restore original weight value
                
                    # Compute numerical gradient
                    numerical_dW_l[i, j] = (plus_loss - minus_loss) / (2 * epsilon)
            numerical_dW.append(numerical_dW_l)
    
        # Compute numerical gradients for biases
        for l in range(len(self.biases)):
            numerical_db_l = np.zeros_like(self.biases[l])
            for i in range(self.biases[l].shape[1]):
                # Perturb bias by epsilon (positive and negative)
                self.biases[l][0, i] += epsilon
                plus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                self.biases[l][0, i] -= 2 * epsilon
                minus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                self.biases[l][0, i] += epsilon  # Restore original bias value
            
                # Compute numerical gradient
                numerical_db_l[0, i] = (plus_loss - minus_loss) / (2 * epsilon)
            numerical_db.append(numerical_db_l)
    
        # Compare the analytical gradients with the numerical gradients
        for l in range(len(dW)):
            dw_diff = np.linalg.norm(dW[l] - numerical_dW[l]) / (np.linalg.norm(dW[l]) + np.linalg.norm(numerical_dW[l]))
            db_diff = np.linalg.norm(db[l] - numerical_db[l]) / (np.linalg.norm(db[l]) + np.linalg.norm(numerical_db[l]))
        
            print(f"Layer {l + 1} - Weight gradient difference: {dw_diff:.7f}")
            print(f"Layer {l + 1} - Bias gradient difference: {db_diff:.7f}")
        
            if dw_diff > 1e-7:
                print(f"Warning: Weight gradient check failed for layer {l + 1}")
            if db_diff > 1e-7:
                print(f"Warning: Bias gradient check failed for layer {l + 1}")
    
        print("Gradient checking complete!")



import numpy as np
import pandas as pd
import wandb

class MLPRegressor:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', learning_rate=0.01, optimizer='sgd', batch_size=32):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_function = activation
        self.optimizer_type = optimizer
        self.batch_size = batch_size
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Define the network layers (input -> hidden layers -> output)
        self.layers = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]) * 0.1)
            self.biases.append(np.zeros((1, self.layers[i+1])))
        
    def activation(self, x):
        if self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)
    
    def activation_derivative(self, x):
        if self.activation_function == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_function == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation_function == 'tanh':
            return 1 - np.tanh(x) ** 2
    
    def forward_propagation(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            a = self.activation(z) if i < len(self.weights) - 1 else z  # No activation on the output layer for regression
            self.layer_outputs.append(a)
        return self.layer_outputs[-1]

    def back_propagation(self, X, Y):
        m = X.shape[0]  # Number of examples
        output_error = (self.layer_outputs[-1] - Y.reshape(-1, 1)**2)/m  # Shape (m, 1)
        deltas = [output_error]  # Start with the output error

        # Backpropagate the error through the hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[-1], self.weights[i+1].T) * self.activation_derivative(self.layer_outputs[i+1])
            deltas.append(delta)

        deltas.reverse()  # Reverse the list of deltas to match the layer order

        # Calculate gradients for weights and biases
        gradients_w = []
        gradients_b = []
        for i in range(len(self.weights)):
            dw = np.dot(self.layer_outputs[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            gradients_w.append(dw)
            gradients_b.append(db)
        
        return gradients_w, gradients_b

    def update_parameters(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def fit(self, X, Y, epochs=1000, validation_data=None, early_stopping=False, patience=10):
        #wandb.init(project="mlp_regression")
        best_val_loss = float('inf')
        wait = 0
        
        for epoch in range(epochs):
            if self.optimizer_type == 'sgd':
                # Shuffle data at each epoch for SGD
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                X, Y = X[indices], Y[indices]
                
                # Stochastic Gradient Descent: process one sample at a time
                for i in range(X.shape[0]):
                    X_single = X[i:i+1]
                    Y_single = Y[i:i+1]
                    
                    # Forward pass
                    predictions = self.forward_propagation(X_single)
                    
                    # Backpropagation
                    gradients_w, gradients_b = self.back_propagation(X_single, Y_single)
                    
                    # Update parameters
                    self.update_parameters(gradients_w, gradients_b)

            elif self.optimizer_type == 'mini_batch':
                # Shuffle data for mini-batch
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                X, Y = X[indices], Y[indices]

                # Mini-batch Gradient Descent: process one mini-batch at a time
                for i in range(0, X.shape[0], self.batch_size):
                    X_batch = X[i:i+self.batch_size]
                    Y_batch = Y[i:i+self.batch_size]
                    
                    # Forward pass
                    predictions = self.forward_propagation(X_batch)
                    
                    # Backpropagation
                    gradients_w, gradients_b = self.back_propagation(X_batch, Y_batch)
                    
                    # Update parameters
                    self.update_parameters(gradients_w, gradients_b)

            elif self.optimizer_type == 'batch':
                # Batch Gradient Descent: process entire dataset at once
                predictions = self.forward_propagation(X)
                
                # Backpropagation
                gradients_w, gradients_b = self.back_propagation(X, Y)
                
                # Update parameters
                self.update_parameters(gradients_w, gradients_b)

            # Compute loss (MSE)
            loss = np.mean((predictions - Y) ** 2)

            # Log metrics to W&B
            val_predictions = self.predict(validation_data[0]) if validation_data else predictions
            val_loss = np.mean((val_predictions - validation_data[1]) ** 2) if validation_data else loss
            rmse = np.sqrt(val_loss)
            r2 = 1 - (np.sum((val_predictions - validation_data[1]) ** 2) / np.sum((validation_data[1] - np.mean(validation_data[1])) ** 2))
            
            wandb.log({"epoch": epoch, "loss": loss, "val_loss": val_loss, "rmse": rmse, "r2": r2})
            
            # Early stopping logic
            if validation_data and early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

    def predict(self, X):
        return self.forward_propagation(X)
    
    def evaluate_metrics(self, Y_true, Y_pred):
        mse = np.mean((Y_pred - Y_true) ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((Y_true - Y_pred) ** 2)
        ss_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        return mse, rmse, r2_score
    
    def gradient_check(self, X, Y, epsilon=1e-7):
        parameters = self.params
        gradients = self.backward_propagation(X, Y, self.forward_propagation(X)[1])
        grad_approx = {}

        for key in parameters:
            grad_approx[key] = np.zeros_like(parameters[key])
            for i in range(parameters[key].shape[0]):
                for j in range(parameters[key].shape[1]):
                    plus_epsilon = parameters[key].copy()
                    plus_epsilon[i, j] += epsilon
                    minus_epsilon = parameters[key].copy()
                    minus_epsilon[i, j] -= epsilon

                    J_plus = self.compute_loss(Y, self.forward_propagation(X)[0])
                    J_minus = self.compute_loss(Y, self.forward_propagation(X)[0])
                    grad_approx[key][i, j] = (J_plus - J_minus) / (2 * epsilon)

        # Compare numerical gradients with backprop gradients
        for key in gradients:
            diff = np.linalg.norm(grad_approx[key] - gradients[key]) / (np.linalg.norm(grad_approx[key]) + np.linalg.norm(gradients[key]))
            print(f'Gradient check for {key}: {diff}')