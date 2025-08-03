import numpy as np
import sys
import os
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MLP')))
from MLPR import MLPRegressor


# class AutoEncoder:
#     def __init__(self, input_size, latent_size, learning_rate=0.001, epochs=500):
#         # Initialize parameters
#         self.input_size = input_size
#         self.latent_size = latent_size
#         self.learning_rate = learning_rate
#         self.epochs = epochs

#         # Initialize weights and biases for the encoder
#         self.W_encoder = np.random.rand(input_size, latent_size) * 0.01  # Weight initialization
#         self.b_encoder = np.zeros((1, latent_size))  # Bias initialization

#         # Initialize weights and biases for the decoder
#         self.W_decoder = np.random.rand(latent_size, input_size) * 0.01  # Weight initialization
#         self.b_decoder = np.zeros((1, input_size))  # Bias initialization

#     def _sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def _sigmoid_derivative(self, x):
#         return x * (1 - x)

#     def fit(self, X):
#         # Train the autoencoder
#         for epoch in range(self.epochs):
#             # Forward pass: Encoder
#             encoded = self._sigmoid(np.dot(X, self.W_encoder) + self.b_encoder)

#             # Forward pass: Decoder
#             decoded = self._sigmoid(np.dot(encoded, self.W_decoder) + self.b_decoder)

#             # Calculate loss (Mean Squared Error)
#             loss = np.mean(np.square(X - decoded))

#             # Backward pass
#             # Calculate the error
#             error = X - decoded
            
#             # Decode the error to get the gradient for the decoder weights
#             d_decoded = error * self._sigmoid_derivative(decoded)

#             # Calculate gradients for the decoder weights and biases
#             self.W_decoder += np.dot(encoded.T, d_decoded) * self.learning_rate
#             self.b_decoder += np.sum(d_decoded, axis=0, keepdims=True) * self.learning_rate

#             # Backpropagate to the encoder
#             encoded_error = np.dot(d_decoded, self.W_decoder.T)
#             d_encoded = encoded_error * self._sigmoid_derivative(encoded)

#             # Calculate gradients for the encoder weights and biases
#             self.W_encoder += np.dot(X.T, d_encoded) * self.learning_rate
#             self.b_encoder += np.sum(d_encoded, axis=0, keepdims=True) * self.learning_rate

#             # Print loss every 50 epochs for monitoring
#             if epoch % 10 == 0:
#                 print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss}')

#     def get_latent(self, X):
#         # Get the reduced representation
#         return self._sigmoid(np.dot(X, self.W_encoder) + self.b_encoder)

#     def reconstruct(self, X):
#         # Reconstruct the input data
#         encoded = self.get_latent(X)
#         return self._sigmoid(np.dot(encoded, self.W_decoder) + self.b_decoder)
    




# class AutoEncoder:
#     def __init__(self, input_size, latent_size, hidden_layers, epochs, learning_rate):
#         self.input_size = input_size
#         self.latent_size = latent_size
#         self.hidden_layers = hidden_layers
#         self.epochs = epochs
#         self.learning_rate = learning_rate
        
#         # # Define the structure for encoder and decoder
#         # encoder_layers = hidden_layers + [self.latent_size]
#         # decoder_layers = list(reversed(hidden_layers))
        
#         # Create an MLP model for both encoder and decoder combined
#         self.model = MLPRegressor(
#             input_size=self.input_size, 
#             hidden_layers=hidden_layers+[self.latent_size]+list(reversed(hidden_layers)),  # Full autoencoder structure
#             output_size=self.input_size, 
#             epochs=self.epochs, 
#             learning_rate=self.learning_rate
#         )
        
#     def train(self, X_train):
#         # Train the autoencoder model by fitting X_train as both inputs and targets (reconstruction)
#         self.model.fit(X_train, X_train)
    
#     def reconstruct(self, X):
#         # Reconstruct the input from the trained model
#         return self.model.predict(X)
    
#     def extract_latent(self, X):
#         # Perform forward propagation and extract the latent layer representation
#         activations = self.model.forward_propagation(X)
#         latent_layer_index = len(self.hidden_layers)  # Latent layer is at the middle of the hidden layers
#         return activations[latent_layer_index]
    


# import numpy as np

# class AutoEncoder:
#     def __init__(self, input_size, hidden_size, latent_size, learning_rate=0.001, epochs=500):
#         # Initialize parameters
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.latent_size = latent_size
#         self.learning_rate = learning_rate
#         self.epochs = epochs

#         # Encoder: Input -> Hidden -> Latent
#         self.W_encoder_1 = np.random.rand(input_size, hidden_size) * 0.01  # Weights for input to hidden layer
#         self.b_encoder_1 = np.zeros((1, hidden_size))  # Bias for hidden layer
        
#         self.W_encoder_2 = np.random.rand(hidden_size, latent_size) * 0.01  # Weights for hidden to latent layer
#         self.b_encoder_2 = np.zeros((1, latent_size))  # Bias for latent layer

#         # Decoder: Latent -> Hidden -> Output
#         self.W_decoder_1 = np.random.rand(latent_size, hidden_size) * 0.01  # Weights for latent to hidden layer
#         self.b_decoder_1 = np.zeros((1, hidden_size))  # Bias for hidden layer
        
#         self.W_decoder_2 = np.random.rand(hidden_size, input_size) * 0.01  # Weights for hidden to output layer
#         self.b_decoder_2 = np.zeros((1, input_size))  # Bias for output layer

#     def _sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def _sigmoid_derivative(self, x):
#         return x * (1 - x)

#     def fit(self, X):
#         # Train the autoencoder
#         for epoch in range(self.epochs):
#             # Forward pass: Encoder
#             hidden_encoded = self._sigmoid(np.dot(X, self.W_encoder_1) + self.b_encoder_1)  # Input -> Hidden
#             latent_encoded = self._sigmoid(np.dot(hidden_encoded, self.W_encoder_2) + self.b_encoder_2)  # Hidden -> Latent

#             # Forward pass: Decoder
#             hidden_decoded = self._sigmoid(np.dot(latent_encoded, self.W_decoder_1) + self.b_decoder_1)  # Latent -> Hidden
#             output_decoded = self._sigmoid(np.dot(hidden_decoded, self.W_decoder_2) + self.b_decoder_2)  # Hidden -> Output

#             # Calculate loss (Mean Squared Error)
#             loss = np.mean(np.square(X - output_decoded))

#             # Backward pass
#             error = X - output_decoded  # Reconstruction error

#             # Decoder gradients
#             d_output_decoded = error * self._sigmoid_derivative(output_decoded)
#             d_hidden_decoded = np.dot(d_output_decoded, self.W_decoder_2.T) * self._sigmoid_derivative(hidden_decoded)
            
#             # Update decoder weights and biases
#             self.W_decoder_2 += np.dot(hidden_decoded.T, d_output_decoded) * self.learning_rate
#             self.b_decoder_2 += np.sum(d_output_decoded, axis=0, keepdims=True) * self.learning_rate
#             self.W_decoder_1 += np.dot(latent_encoded.T, d_hidden_decoded) * self.learning_rate
#             self.b_decoder_1 += np.sum(d_hidden_decoded, axis=0, keepdims=True) * self.learning_rate

#             # Encoder gradients
#             d_latent_encoded = np.dot(d_hidden_decoded, self.W_decoder_1.T) * self._sigmoid_derivative(latent_encoded)
#             d_hidden_encoded = np.dot(d_latent_encoded, self.W_encoder_2.T) * self._sigmoid_derivative(hidden_encoded)

#             # Update encoder weights and biases
#             self.W_encoder_2 += np.dot(hidden_encoded.T, d_latent_encoded) * self.learning_rate
#             self.b_encoder_2 += np.sum(d_latent_encoded, axis=0, keepdims=True) * self.learning_rate
#             self.W_encoder_1 += np.dot(X.T, d_hidden_encoded) * self.learning_rate
#             self.b_encoder_1 += np.sum(d_hidden_encoded, axis=0, keepdims=True) * self.learning_rate

#             # Print loss every 50 epochs for monitoring
#             if epoch % 10 == 0:
#                 print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss}')

#     def get_latent(self, X):
#         # Get the reduced representation (latent space)
#         hidden_encoded = self._sigmoid(np.dot(X, self.W_encoder_1) + self.b_encoder_1)
#         latent_encoded = self._sigmoid(np.dot(hidden_encoded, self.W_encoder_2) + self.b_encoder_2)
#         return latent_encoded

#     def reconstruct(self, X):
#         # Reconstruct the input data from latent space
#         latent_encoded = self.get_latent(X)
#         hidden_decoded = self._sigmoid(np.dot(latent_encoded, self.W_decoder_1) + self.b_decoder_1)
#         output_decoded = self._sigmoid(np.dot(hidden_decoded, self.W_decoder_2) + self.b_decoder_2)
#         return output_decoded
    



# import numpy as np

# class AutoEncoder:
#     def __init__(self, input_size, hidden_size, latent_size, learning_rate=0.0001, epochs=100):
#         # Initialize parameters
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.latent_size = latent_size
#         self.learning_rate = learning_rate
#         self.epochs = epochs

#         # Encoder: Input -> Hidden -> Latent
#         self.W_encoder_1 = np.random.rand(input_size, hidden_size) * 0.01
#         self.b_encoder_1 = np.zeros((1, hidden_size))
        
#         self.W_encoder_2 = np.random.rand(hidden_size, latent_size) * 0.01
#         self.b_encoder_2 = np.zeros((1, latent_size))

#         # Decoder: Latent -> Hidden -> Output
#         self.W_decoder_1 = np.random.rand(latent_size, hidden_size) * 0.01
#         self.b_decoder_1 = np.zeros((1, hidden_size))
        
#         self.W_decoder_2 = np.random.rand(hidden_size, input_size) * 0.01
#         self.b_decoder_2 = np.zeros((1, input_size))

#     def _tanh(self, x):
#         return np.tanh(x)

#     def _tanh_derivative(self, x):
#         return 1 - np.square(np.tanh(x))

#     def fit(self, X):
#         for epoch in range(self.epochs):
#             # Forward pass: Encoder
#             hidden_encoded = self._tanh(np.dot(X, self.W_encoder_1) + self.b_encoder_1)
#             latent_encoded = self._tanh(np.dot(hidden_encoded, self.W_encoder_2) + self.b_encoder_2)

#             # Forward pass: Decoder
#             hidden_decoded = self._tanh(np.dot(latent_encoded, self.W_decoder_1) + self.b_decoder_1)
#             output_decoded = self._tanh(np.dot(hidden_decoded, self.W_decoder_2) + self.b_decoder_2)

#             # Calculate loss (Mean Squared Error)
#             loss = np.mean(np.square(X - output_decoded))

#             # Backward pass
#             error = X - output_decoded

#             # Decoder gradients
#             d_output_decoded = error * self._tanh_derivative(output_decoded)
#             d_hidden_decoded = np.dot(d_output_decoded, self.W_decoder_2.T) * self._tanh_derivative(hidden_decoded)
            
#             self.W_decoder_2 += np.dot(hidden_decoded.T, d_output_decoded) * self.learning_rate
#             self.b_decoder_2 += np.sum(d_output_decoded, axis=0, keepdims=True) * self.learning_rate
#             self.W_decoder_1 += np.dot(latent_encoded.T, d_hidden_decoded) * self.learning_rate
#             self.b_decoder_1 += np.sum(d_hidden_decoded, axis=0, keepdims=True) * self.learning_rate

#             # Encoder gradients
#             d_latent_encoded = np.dot(d_hidden_decoded, self.W_decoder_1.T) * self._tanh_derivative(latent_encoded)
#             d_hidden_encoded = np.dot(d_latent_encoded, self.W_encoder_2.T) * self._tanh_derivative(hidden_encoded)

#             self.W_encoder_2 += np.dot(hidden_encoded.T, d_latent_encoded) * self.learning_rate
#             self.b_encoder_2 += np.sum(d_latent_encoded, axis=0, keepdims=True) * self.learning_rate
#             self.W_encoder_1 += np.dot(X.T, d_hidden_encoded) * self.learning_rate
#             self.b_encoder_1 += np.sum(d_hidden_encoded, axis=0, keepdims=True) * self.learning_rate

#             if epoch % 10 == 0:
#                 print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss}')
#     def get_latent(self, X):
#         # Get the reduced representation
#         hidden_encoded = self._tanh(np.dot(X, self.W_encoder_1) + self.b_encoder_1)
#         return self._tanh(np.dot(hidden_encoded, self.W_encoder_2) + self.b_encoder_2)

#     def reconstruct(self, X):
#         # Reconstruct the input data
#         latent_encoded = self.get_latent(X)
#         hidden_decoded = self._tanh(np.dot(latent_encoded, self.W_decoder_1) + self.b_decoder_1)
#         return self._tanh(np.dot(hidden_decoded, self.W_decoder_2) + self.b_decoder_2)
    




import numpy as np

class AutoEncoder:
    def __init__(self, input_size, hidden_size, latent_size, learning_rate=0.001, epochs=500):
        # Initialize parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases for the encoder
        self.W_encoder = np.random.randn(input_size, hidden_size) * 0.01  # Small random weights
        self.b_encoder = np.zeros((1, hidden_size))  # Bias initialization

        # Initialize weights and biases for the latent layer
        self.W_latent = np.random.randn(hidden_size, latent_size) * 0.01
        self.b_latent = np.zeros((1, latent_size))

        # Initialize weights and biases for the decoder
        self.W_decoder = np.random.randn(latent_size, hidden_size) * 0.01
        self.b_decoder = np.zeros((1, hidden_size))

        # Output layer weights
        self.W_output = np.random.randn(hidden_size, input_size) * 0.01
        self.b_output = np.zeros((1, input_size))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def fit(self, X):
        # Train the autoencoder
        for epoch in range(self.epochs):
            # Forward pass: Encoder
            hidden = self._relu(np.dot(X, self.W_encoder) + self.b_encoder)
            latent = self._sigmoid(np.dot(hidden, self.W_latent) + self.b_latent)

            # Forward pass: Decoder
            hidden_reconstructed = self._relu(np.dot(latent, self.W_decoder) + self.b_decoder)
            decoded = self._sigmoid(np.dot(hidden_reconstructed, self.W_output) + self.b_output)

            # Calculate loss (Mean Squared Error)
            loss = np.mean(np.square(X - decoded))

            # Backward pass
            # Calculate the error
            error = X - decoded

            # Backpropagate through the output layer
            d_output = error * self._sigmoid_derivative(decoded)
            self.W_output += np.dot(hidden_reconstructed.T, d_output) * self.learning_rate
            self.b_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

            # Backpropagate to the hidden layer
            hidden_error = np.dot(d_output, self.W_output.T)
            d_hidden_reconstructed = hidden_error * self._relu_derivative(hidden_reconstructed)
            self.W_decoder += np.dot(latent.T, d_hidden_reconstructed) * self.learning_rate
            self.b_decoder += np.sum(d_hidden_reconstructed, axis=0, keepdims=True) * self.learning_rate

            # Backpropagate to the latent layer
            hidden_error_latent = np.dot(d_hidden_reconstructed, self.W_decoder.T)
            d_latent = hidden_error_latent * self._sigmoid_derivative(latent)
            self.W_latent += np.dot(hidden.T, d_latent) * self.learning_rate
            self.b_latent += np.sum(d_latent, axis=0, keepdims=True) * self.learning_rate

            # Backpropagate to the encoder
            encoder_error = np.dot(d_latent, self.W_latent.T)
            d_hidden = encoder_error * self._relu_derivative(hidden)
            self.W_encoder += np.dot(X.T, d_hidden) * self.learning_rate
            self.b_encoder += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

            # Print loss every 50 epochs for monitoring
            if epoch % 1 == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}')

    def get_latent(self, X):
        # Get the reduced representation
        hidden = self._relu(np.dot(X, self.W_encoder) + self.b_encoder)
        return self._sigmoid(np.dot(hidden, self.W_latent) + self.b_latent)

    def reconstruct(self, X):
        # Reconstruct the input data
        latent = self.get_latent(X)
        hidden_reconstructed = self._relu(np.dot(latent, self.W_decoder) + self.b_decoder)
        return self._sigmoid(np.dot(hidden_reconstructed, self.W_output) + self.b_output)