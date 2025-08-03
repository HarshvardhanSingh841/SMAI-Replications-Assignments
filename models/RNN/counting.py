import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# class CountingRNN(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, output_size=1, dropout_prob=0.3):
#         super(CountingRNN, self).__init__()
        
#         # RNN layer (GRU or RNN)
#         self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        
#         # Dropout layer to prevent overfitting
#         self.dropout = nn.Dropout(dropout_prob)
        
#         # Fully connected layer to map RNN output to final count prediction
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         # Passing the input through the RNN
#         out, _ = self.rnn(x)
        
#         # Apply dropout for regularization
#         out = self.dropout(out)
        
#         # Use the last hidden state for classification (last time step)
#         out = self.fc(out[:, -1, :])
        
#         return out


class CountingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        """
        Initializes the RNN model.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Number of hidden units.
            output_size (int): Size of the output features.
            num_layers (int): Number of RNN layers.
            dropout (float): Dropout probability.
        """
        super(CountingRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, seq_lengths):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Padded input tensor of shape (batch_size, max_seq_len, input_size).
            seq_lengths (Tensor): Actual lengths of each sequence.

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size).
        """
        packed_x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        final_output = output[range(output.size(0)), seq_lengths - 1]  # Gather outputs at the last valid time step
        return self.fc(final_output)