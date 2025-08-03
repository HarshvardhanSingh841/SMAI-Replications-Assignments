import numpy as np
import wandb

class MLPRegressor:
    def __init__(self, input_size, output_size, hidden_layers=[64, 64], learning_rate=0.01, 
                 activation='relu', optimizer='sgd', batch_size=32, epochs=100 ,):
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
            
            # For regression, no softmax at output layer, just apply the linear activation
            if i == len(self.weights) - 1:
                A = Z  # Linear activation at output layer
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
        dA = activations[-1] - Y  # Loss derivative w.r.t final activation (for regression, MSE)

        for i in reversed(range(len(self.weights))):
            dZ = dA if i == len(self.weights) - 1 else dA * self.activation_derivative(activations[i+1], self.activation)
            dW_i = np.dot(activations[i].T, dZ) / m
            db_i = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.weights[i].T)
        
            dW.insert(0, dW_i)
            db.insert(0, db_i)

        return dW, db
    def backward_propagation(self, X, Y, activations, Zs):
        m = X.shape[0]  # Number of samples
        dW = []
        db = []
        
        # Compute the derivative of loss with respect to the final activation (MSE Loss for regression)
        dA = activations[-1] - Y  # Shape: (m, 1) for regression output

        # Loop over layers in reverse order
        for i in reversed(range(len(self.weights))):
            # For the output layer, we directly use dA, no need to multiply by derivative of activation function
            if i == len(self.weights) - 1:  
                dZ = dA  # Shape: (m, 1) (no need for activation derivative for output in MSE)
            else:
                # For hidden layers, calculate dZ using activation derivative
                dZ = dA * self.activation_derivative(activations[i+1], self.activation)  # Shape: (m, units_in_layer)

            # Compute gradients for weights and biases
            dW_i = np.dot(activations[i].T, dZ) / m  # Shape: (units_in_previous_layer, units_in_layer)
            db_i = np.sum(dZ, axis=0, keepdims=True) / m  # Shape: (1, units_in_layer)
            
            # Backpropagate the gradient to the previous layer
            dA = np.dot(dZ, self.weights[i].T)  # Shape: (m, units_in_previous_layer)

            # Store the gradients for this layer
            dW.insert(0, dW_i)  # Insert at the beginning to maintain correct order
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

    def fit(self, X, Y, X_val=None, Y_val=None, early_stopping=False, patience=5):
        best_loss = float('inf')
        patience_counter = 0

        if self.optimizer in ['batch', 'mini_batch']:
            num_batches = X.shape[0] // self.batch_size + (1 if X.shape[0] % self.batch_size != 0 else 0)

        for epoch in range(self.epochs):
            if self.optimizer == 'sgd':
                for i in range(X.shape[0]):
                    activations, Zs = self.forward_propagation(X[i:i+1])  # Single example
                    dW, db = self.backward_propagation(X[i:i+1], Y[i:i+1], activations, Zs)
                    self.update_parameters(dW, db)

            elif self.optimizer in ['mini_batch', 'batch']:
                activations, Zs = self.forward_propagation(X)
                dW, db = self.backward_propagation(X, Y, activations, Zs)

                if self.optimizer == 'mini_batch':
                    for batch_index in range(num_batches):
                        start = batch_index * self.batch_size
                        end = start + self.batch_size
                        X_batch = X[start:end]
                        Y_batch = Y[start:end]
                        
                        activations_batch, Zs_batch = self.forward_propagation(X_batch)
                        dW_batch, db_batch = self.backward_propagation(X_batch, Y_batch, activations_batch, Zs_batch)
                        self.update_parameters(dW_batch, db_batch)

                elif self.optimizer == 'batch':
                    self.update_parameters(dW, db)

            train_loss = self.calculate_loss(Y, activations[-1])
            wandb.log({'Epoch': epoch + 1, 'Train Loss': train_loss})

            if X_val is not None and Y_val is not None:
                val_activations, _ = self.forward_propagation(X_val)
                val_loss = self.calculate_loss(Y_val, val_activations[-1])
                wandb.log({'Validation Loss': val_loss})

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
        loss = np.mean((Y_true - Y_pred) ** 2)  # Mean Squared Error (MSE)
        return loss

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]  # Direct output for regression

    def gradient_check(self, X, Y, epsilon=1e-7):
        activations, Zs = self.forward_propagation(X)
        dW, db = self.backward_propagation(X, Y, activations, Zs)

        numerical_dW, numerical_db = [], []

        for l in range(len(self.weights)):
            numerical_dW_l = np.zeros_like(self.weights[l])
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    self.weights[l][i, j] += epsilon
                    plus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                    self.weights[l][i, j] -= 2 * epsilon
                    minus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                    self.weights[l][i, j] += epsilon
                    
                    numerical_dW_l[i, j] = (plus_loss - minus_loss) / (2 * epsilon)
            numerical_dW.append(numerical_dW_l)

        for l in range(len(self.biases)):
            numerical_db_l = np.zeros_like(self.biases[l])
            for i in range(self.biases[l].shape[1]):
                self.biases[l][0, i] += epsilon
                plus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                self.biases[l][0, i] -= 2 * epsilon
                minus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                self.biases[l][0, i] += epsilon
                
                numerical_db_l[0, i] = (plus_loss - minus_loss) / (2 * epsilon)
            numerical_db.append(numerical_db_l)

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




















#logistic regression
class MLPLogistic:
    def __init__(self, input_size, output_size, hidden_layers=[64, 64], learning_rate=0.01, 
                 activation='relu', optimizer='sgd', batch_size=32, epochs=100 , loss_function = 'mse'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_function = 'mse'

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialize_parameters()

    def initialize_parameters(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i+1])))

    # def forward_propagation(self, X):
    #     activations = [X]  # List to store activations layer by layer
    #     Zs = []  # List to store linear transformations Z = W.X + b

    #     for i in range(len(self.weights)):
    #         Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
    #         Zs.append(Z)
            
    #         # For regression, no softmax at output layer, just apply the linear activation
    #         if i == len(self.weights) - 1:
    #             A = 1 / (1 + np.exp(-Z)) #sigmoid activation at output layer
    #         else:
    #             A = self.activate(Z, self.activation)  # Hidden layers use the selected activation function
    #         activations.append(A)

    #     return activations, Zs
    def forward_propagation(self, X):
        Z = np.dot(X, self.weights[0]) + self.biases[0]  # For the output layer
        A = self.activate(Z,self.activation)  # Sigmoid activation for output
        return A , Z
    def activate(self, Z, activation_function):
        if activation_function == 'relu':
            return np.maximum(0, Z)
        elif activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation_function == 'tanh':
            return np.tanh(Z)
        elif activation_function == 'linear':
            return Z

    # def backward_propagation(self, X, Y, activations, Zs):
    #     m = X.shape[0]  
    #     dW = []
    #     db = []
    #     dA = activations[-1] - Y  # Loss derivative w.r.t final activation (for regression, MSE)

    #     for i in reversed(range(len(self.weights))):
    #         dZ = dA if i == len(self.weights) - 1 else dA * self.activation_derivative(activations[i+1], self.activation)
    #         dW_i = np.dot(activations[i].T, dZ) / m
    #         db_i = np.sum(dZ, axis=0, keepdims=True) / m
    #         dA = np.dot(dZ, self.weights[i].T)
        
    #         dW.insert(0, dW_i)
    #         db.insert(0, db_i)

    #     return dW, db
    # def backward_propagation(self, X, Y, activations, Zs):
    #     m = X.shape[0]  # Number of samples
    #     dW = []
    #     db = []
        
    #     # # Compute the derivative of loss with respect to the final activation (MSE Loss for regression)
    #     # dA = activations[-1] - Y  # Shape: (m, 1) for regression output

    #     if self.loss_function == 'bce':
    #         dA = (activations[-1] - Y) / (activations[-1] * (1 - activations[-1]))  # BCE gradient  # Gradient for BCE loss
    #     elif self.loss_function == 'mse':
    #         dA = activations[-1] - Y  # Gradient for MSE loss


    #     # Loop over layers in reverse order
    #     for i in reversed(range(len(self.weights))):
    #         # For the output layer, we directly use dA, no need to multiply by derivative of activation function
    #         if i == len(self.weights) - 1:  
    #             dZ = dA  # Shape: (m, 1) (no need for activation derivative for output in MSE)
    #         else:
    #             # For hidden layers, calculate dZ using activation derivative
    #             dZ = dA * self.activation_derivative(activations[i+1], self.activation)  # Shape: (m, units_in_layer)

    #         # Compute gradients for weights and biases
    #         dW_i = np.dot(activations[i].T, dZ) / m  # Shape: (units_in_previous_layer, units_in_layer)
    #         db_i = np.sum(dZ, axis=0, keepdims=True) / m  # Shape: (1, units_in_layer)
            
    #         # Backpropagate the gradient to the previous layer
    #         dA = np.dot(dZ, self.weights[i].T)  # Shape: (m, units_in_previous_layer)

    #         # Store the gradients for this layer
    #         dW.insert(0, dW_i)  # Insert at the beginning to maintain correct order
    #         db.insert(0, db_i)

    #     return dW, db

    def backward_propagation(self, X, Y, activations, Zs):
        m = X.shape[0]  # Number of samples
        dW = []
        db = []
          # Initialize dA for debugging
        #dA = None
        # print("Current loss function:", self.loss_function)
        # Calculate dA based on the chosen loss function
        if self.loss_function == 'mse':
            dA = activations - Y  # MSE gradient
        elif self.loss_function == 'bce':
            dA = (activations - Y) / (activations * (1 - activations))  # BCE gradient
        else:
            raise ValueError("Invalid loss function specified! Must be 'mse' or 'bce'.")
        # Debugging statement to check the value of dA
        # if dA is None:
        #     raise ValueError("dA is not initialized! Check loss function assignment.")
        # For a model with no hidden layers, there's only one weight layer
        dZ = dA  # The output layer's derivative is directly dA

        # Compute gradients for weights and biases
        dW_i = np.dot(X.T, dZ) / m  # Shape: (num_features, 1)
        db_i = np.sum(dZ, axis=0, keepdims=True) / m  # Shape: (1, 1)

        dW.append(dW_i)
        db.append(db_i)

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

    def fit(self, X, Y, X_val=None, Y_val=None, early_stopping=False, patience=5 , loss_function = 'mse'):
        best_loss = float('inf')
        patience_counter = 0
        self.loss_function = loss_function
        # Set the appropriate loss function based on the parameter
        if loss_function == 'bce':
            loss_method = self.binary_cross_entropy_loss
        elif loss_function == 'mse':
            loss_method = self.calculate_loss

        # if loss_function is not None:
        #     self.loss_function = loss_function  # Should be a string like 'bce'
        #     print(f"Current loss function: {self.loss_function}")  


        if self.optimizer in ['batch', 'mini_batch']:
            num_batches = X.shape[0] // self.batch_size + (1 if X.shape[0] % self.batch_size != 0 else 0)

        for epoch in range(self.epochs):
            if self.optimizer == 'sgd':
                for i in range(X.shape[0]):
                    activations, Zs = self.forward_propagation(X[i:i+1])  # Single example
                    dW, db = self.backward_propagation(X[i:i+1], Y[i:i+1], activations, Zs)
                    self.update_parameters(dW, db)

            elif self.optimizer in ['mini_batch', 'batch']:
                activations, Zs = self.forward_propagation(X)
                dW, db = self.backward_propagation(X, Y, activations, Zs)

                if self.optimizer == 'mini_batch':
                    for batch_index in range(num_batches):
                        start = batch_index * self.batch_size
                        end = start + self.batch_size
                        X_batch = X[start:end]
                        Y_batch = Y[start:end]
                        
                        activations_batch, Zs_batch = self.forward_propagation(X_batch)
                        dW_batch, db_batch = self.backward_propagation(X_batch, Y_batch, activations_batch, Zs_batch)
                        self.update_parameters(dW_batch, db_batch)

                elif self.optimizer == 'batch':
                    self.update_parameters(dW, db)

            train_loss = loss_method(Y, self.forward_propagation(X)[0])
            wandb.log({'Epoch': epoch + 1, 'Train Loss': train_loss})

            if X_val is not None and Y_val is not None:
                val_activations, _ = self.forward_propagation(X_val)
                val_loss = loss_method(Y_val, val_activations)
                wandb.log({'Validation Loss': val_loss})

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

    def binary_cross_entropy_loss(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        
        loss = -np.mean(Y_true * np.log(Y_pred + 1e-15) + (1 - Y_true) * np.log(1 - Y_pred + 1e-15))
        return loss
    def calculate_loss(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        #print(Y_pred.shape[0])
        loss = np.mean((Y_true - Y_pred) ** 2)  # Mean Squared Error (MSE)
        return loss

    def predict(self, X):
        activations, _ = self.forward_propagation(X)

        return activations.flatten()  # Direct output for regression

    def gradient_check(self, X, Y, epsilon=1e-7):
        activations, Zs = self.forward_propagation(X)
        dW, db = self.backward_propagation(X, Y, activations, Zs)

        numerical_dW, numerical_db = [], []

        for l in range(len(self.weights)):
            numerical_dW_l = np.zeros_like(self.weights[l])
            for i in range(self.weights[l].shape[0]):
                for j in range(self.weights[l].shape[1]):
                    self.weights[l][i, j] += epsilon
                    plus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                    self.weights[l][i, j] -= 2 * epsilon
                    minus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                    self.weights[l][i, j] += epsilon
                    
                    numerical_dW_l[i, j] = (plus_loss - minus_loss) / (2 * epsilon)
            numerical_dW.append(numerical_dW_l)

        for l in range(len(self.biases)):
            numerical_db_l = np.zeros_like(self.biases[l])
            for i in range(self.biases[l].shape[1]):
                self.biases[l][0, i] += epsilon
                plus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                self.biases[l][0, i] -= 2 * epsilon
                minus_loss = self.calculate_loss(Y, self.forward_propagation(X)[0][-1])
                self.biases[l][0, i] += epsilon
                
                numerical_db_l[0, i] = (plus_loss - minus_loss) / (2 * epsilon)
            numerical_db.append(numerical_db_l)

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