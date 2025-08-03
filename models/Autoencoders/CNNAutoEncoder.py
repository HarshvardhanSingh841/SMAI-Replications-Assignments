import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class CnnAutoencoder(nn.Module):
    def __init__(self, latent_dim=32, kernel_size=3, filters=[32, 64]):
        super(CnnAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, filters[0], kernel_size=kernel_size, stride=2, padding=1),  # Output: [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[1], kernel_size=kernel_size, stride=2, padding=1),  # Output: [B, 64, 7, 7]
            nn.ReLU(),
            nn.Flatten(),  # Flatten to [B, 64*7*7] for the linear layer
            nn.Linear(filters[1]*7*7, latent_dim),  # Output: [B, latent_dim]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, filters[1]*7*7),  # Expand back to shape [B, 64*7*7]
            nn.Unflatten(1, (filters[1], 7, 7)),  # Output shape: [B, 64, 7, 7]
            nn.ConvTranspose2d(filters[1], filters[0], kernel_size=kernel_size, stride=2, padding=1, output_padding=1),  # Output: [B, 32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(filters[0], 1, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),  # Output: [B, 1, 28, 28]
            nn.Sigmoid()  # Output range [0, 1]
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed
    



class CnnAuto(nn.Module):
    def __init__(self, latent_dim, num_layers):
        super(CnnAuto, self).__init__()
        self.encoder = self.build_encoder(latent_dim, num_layers)
        self.decoder = self.build_decoder(latent_dim, num_layers)

    def build_encoder(self, latent_dim, num_layers):
        layers = []
        in_channels = 1  # For grayscale images
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, 32, kernel_size=(7, 7), stride=2, padding=3))
            layers.append(nn.ReLU())
            in_channels = 32  # Output from previous layer
        # Create a dummy input to calculate the output size after the encoder
        dummy_input = torch.zeros(1, 1, 128, 128)
        output_size = self.get_output_size(layers, dummy_input)
        layers.append(nn.Flatten())
        layers.append(nn.Linear(output_size, latent_dim))  # Adjust based on your output shape
        return nn.Sequential(*layers)

    def get_output_size(self, layers, dummy_input):
        with torch.no_grad():
            for layer in layers:
                dummy_input = layer(dummy_input)
        return dummy_input.numel()  # Returns the total number of elements

    def build_decoder(self, latent_dim, num_layers):
        layers = []
        layers.append(nn.Linear(latent_dim, 32 * 16 * 16))  # Adjust based on your output shape from encoder
        layers.append(nn.ReLU())
        layers.append(nn.Unflatten(1, (32, 16, 16)))  # Reshape for deconvolution
        for _ in range(num_layers):
            layers.append(nn.ConvTranspose2d(32, 32, kernel_size=(7, 7), stride=2, padding=3))
            layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(32, 1, kernel_size=(7, 7), stride=1, padding=3))  # Final layer to output shape
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)  # Pass the input through the encoder
        x_reconstructed = self.decoder(z)  # Pass the encoded output through the decoder
        return x_reconstructed  # Return the reconstructed output
