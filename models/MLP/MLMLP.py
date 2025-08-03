import numpy as np
import wandb

class MLPMultiLabelClassifier:
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

    def initialize_parameters(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i + 1])))

    def forward_propagation(self, X):
        activations = [X]  # List to store activations layer by layer
        Zs = []  # List to store linear transformations Z = W.X + b

        for i in range(len(self.weights)):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            Zs.append(Z)

            if i == len(self.weights) - 1:
                A = self.sigmoid(Z)  # Apply sigmoid to the output layer for multi-label
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
        dA = activations[-1] - Y  # Loss derivative for multi-label classification

        for i in reversed(range(len(self.weights))):
            dZ = dA * self.activation_derivative(activations[i + 1], self.activation)
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
            return np.ones_like(A)  # Derivative of linear is 1

    def fit(self, X, Y, X_val=None, Y_val=None, early_stopping=False, patience=5):
        best_loss = float('inf')
        patience_counter = 0

        # For batch or mini-batch gradient descent
        if self.optimizer in ['batch', 'mini_batch']:
            num_batches = X.shape[0] // self.batch_size + (1 if X.shape[0] % self.batch_size != 0 else 0)

        for epoch in range(self.epochs):
            if self.optimizer == 'sgd':
                for i in range(X.shape[0]):
                    # Forward and backward propagation for each example
                    activations, Zs = self.forward_propagation(X[i:i + 1])  # Single example
                    dW, db = self.backward_propagation(X[i:i + 1], Y[i:i + 1], activations, Zs)
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

                        # Forward and backward propagation for the mini-batch
                        activations_batch, Zs_batch = self.forward_propagation(X_batch)
                        dW_batch, db_batch = self.backward_propagation(X_batch, Y_batch, activations_batch, Zs_batch)

                        # Update parameters for mini-batch
                        self.update_parameters(dW_batch, db_batch)

                elif self.optimizer == 'batch':
                    self.update_parameters(dW, db)

            # Logging and validation checks
            train_loss = self.calculate_loss(Y, activations[-1])
            wandb.log({'Epoch': epoch + 1, 'Train Loss': train_loss})

            train_accuracy = self.calculate_accuracy(Y, activations[-1])
            wandb.log({'Train Accuracy': train_accuracy})

            if X_val is not None and Y_val is not None:
                val_activations, _ = self.forward_propagation(X_val)
                val_loss = self.calculate_loss(Y_val, val_activations[-1])
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
        loss = -np.sum(Y_true * np.log(Y_pred_clipped) + (1 - Y_true) * np.log(1 - Y_pred_clipped)) / m
        return loss

    def calculate_accuracy(self, Y_true, Y_pred):
        Y_pred_decoded = self.decode_predictions(Y_pred)
        accuracy = np.sum(np.all(Y_pred_decoded == Y_true, axis=1)) / Y_true.shape[0]
        return accuracy

    def calculate_metrics(self, Y_true, Y_pred):
        # Calculate precision, recall, F1-score, and Hamming loss for multi-label
        tp = np.sum((Y_true == 1) & (Y_pred == 1), axis=0)
        fp = np.sum((Y_true == 0) & (Y_pred == 1), axis=0)
        fn = np.sum((Y_true == 1) & (Y_pred == 0), axis=0)

        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)

        # Hamming loss calculation
        hamming_loss = np.mean(np.not_equal(Y_true, Y_pred).astype(float))

        return precision, recall, f1_score, hamming_loss

    def sigmoid(self, Z):
        Z = np.asarray(Z)
        return 1 / (1 + np.exp(-Z))  # Sigmoid for multi-label outputs

    def decode_predictions(self, Y_pred):
        """
        Decode the predictions from the output layer.
        """
        return (Y_pred > 0.5).astype(int)  # Threshold to convert probabilities to binary predictions