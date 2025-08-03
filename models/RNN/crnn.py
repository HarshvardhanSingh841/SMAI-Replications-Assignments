import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
IMAGE_SIZE = (256,64)
# class CRNN(nn.Module):
#     def __init__(self, input_channels, hidden_size, output_size, num_layers=1, dropout=0.5):
#         super(CRNN, self).__init__()

#         # CNN Encoder: Simplified to one layer
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),  # Single Conv layer
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool to reduce spatial size
#             nn.Dropout(dropout),
#         )

#         # Corrected flattened size after CNN layers
#         self.cnn_out_size = 64 * (IMAGE_SIZE[0] // 2) * (IMAGE_SIZE[1] // 2)  # e.g., 262144

#         # RNN Decoder: Process the extracted features sequentially
#         self.rnn = nn.RNN(
#             input_size=self.cnn_out_size,  # Correct input size for RNN
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout,
#             batch_first=True
#         )

#         # Output layer: Predict the character (or label) sequence
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x, lengths):
#         # Forward pass through CNN encoder
#         x = self.cnn(x)

#         # Print shape after CNN layers for debugging
#         print("Shape after CNN layers:", x.shape)

#         # Flatten the spatial dimensions (height, width, channels)
#         x = x.view(x.size(0), -1)  # Flatten to [batch_size, feature_size]
        
#         # Reshape into [batch_size, sequence_length, feature_size] for RNN
#         sequence_length = 1  # Since each image is treated as one "time step"
#         x = x.view(x.size(0), sequence_length, -1)

#         # Print shape before RNN for debugging
#         print("Shape before RNN:", x.shape)

#         # Pack the padded sequences for RNN (for variable-length target sequences)
#         packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

#         # Forward pass through RNN decoder
#         packed_output, _ = self.rnn(packed_input)

#         # Unpack the output
#         output, _ = pad_packed_sequence(packed_output, batch_first=True)

#         # Predict the character/label at each time step
#         output = self.fc(output)

#         return output

class CRNN(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(CRNN, self).__init__()

        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),  # Single Conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool to reduce spatial size
            nn.Dropout(dropout),
        )

        # Output size of CNN after pooling
        self.cnn_out_size = 64 * (IMAGE_SIZE[0] // 2) * (IMAGE_SIZE[1] // 2)  # Example: 262144

        # RNN Decoder: Process the extracted features sequentially
        self.rnn = nn.RNN(
            input_size=self.cnn_out_size,  # Correct input size for RNN
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Output layer: Predict the character (or label) sequence
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through CNN encoder
        x = self.cnn(x)

        # Flatten the CNN output
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, feature_size]
        
        # Reshape into [batch_size, sequence_length, feature_size] for RNN
        sequence_length = 1  # Treat the whole image as a single time step for now
        x = x.view(x.size(0), sequence_length, -1)

        # Forward pass through RNN decoder
        rnn_output, _ = self.rnn(x)

        # Apply the fully connected layer to get the final output
        output = self.fc(rnn_output)

        return output


import torch
import torch.nn as nn
import torch.nn.functional as F

class OCRModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=256):
        super(OCRModel, self).__init__()
        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # RNN Decoder
        self.rnn = nn.GRU(
            input_size=64 * (256 // 4),  # Feature vector size from CNN
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Pass through CNN
        batch_size = x.size(0)
        features = self.cnn(x)  # Output shape: [B, 64, H/4, W/4]
        features = features.permute(0, 2, 3, 1)  # Shape: [B, H/4, W/4, 64]
        features = features.reshape(batch_size, -1, 64 * (256 // 4))  # Shape: [B, Seq_len, Feature_dim]

        # Pass through RNN
        rnn_out, _ = self.rnn(features)  # Shape: [B, Seq_len, Hidden_dim]
        output = self.fc(rnn_out)  # Shape: [B, Seq_len, Num_classes]

        return output