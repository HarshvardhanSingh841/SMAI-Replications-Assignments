import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/RNN')))
from counting import CountingRNN
# Function to generate binary sequences and their labels
def generate_dataset(num_samples, max_length):
    sequences = []
    labels = []
    
    for _ in range(num_samples):
        length = np.random.randint(1, max_length + 1)  # Random sequence length between 1 and max_length
        sequence = np.random.randint(0, 2, size=length)  # Binary sequence (0s and 1s)
        count_ones = np.sum(sequence)  # Count of 1's in the sequence
        sequences.append(sequence)
        labels.append(count_ones)
    
    return sequences, labels

# Generate 100k samples
# Generate data
max_seq_len = 16
sequences, labels = generate_dataset(100000, max_seq_len)
X_train, X_temp, y_train, y_temp = train_test_split(sequences, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


for i in range(5):
    print(f"Sequence {i+1}: {X_train[i]}")
    print(f"Label {i+1}: {y_train[i]}")
    print()  # Adding a blank line for readability

# Print the dataset sizes
print(f"Training size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")




class BinarySequenceDataset(Dataset):
    def __init__(self, sequences, labels, max_seq_len=16):
        """
        Initializes the dataset with sequences and labels.

        Args:
            sequences (list of lists): List of binary sequences.
            labels (list): List of corresponding counts of '1's in each sequence.
            max_seq_len (int): Maximum sequence length for padding.
        """
        self.sequences = sequences
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple (sequence_tensor, label_tensor, seq_length):
                - sequence_tensor: Padded sequence tensor of shape (max_seq_len, 1).
                - label_tensor: Label tensor (float).
                - seq_length: Original length of the sequence.
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        seq_len = len(sequence)
        
        # Ensure sequence is a list before padding
        sequence = list(sequence)
        padded_sequence = sequence + [0] * (self.max_seq_len - seq_len)
        
        sequence_tensor = torch.tensor(padded_sequence, dtype=torch.float32).unsqueeze(-1)  # Shape: (max_seq_len, 1)
        label_tensor = torch.tensor(label, dtype=torch.float32)  # Scalar label
        return sequence_tensor, label_tensor, seq_len


# Hyperparameters
max_seq_len = 16
input_size = 1
hidden_size = 64
output_size = 1
num_layers = 2
dropout_prob = 0.2
batch_size = 64
num_epochs = 10
learning_rate = 0.001
# Create Dataset and DataLoader
train_dataset = BinarySequenceDataset(X_train, y_train, max_seq_len)
val_dataset = BinarySequenceDataset(X_val, y_val, max_seq_len)
test_dataset = BinarySequenceDataset(X_test, y_test, max_seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = CountingRNN(input_size, hidden_size, output_size, num_layers, dropout_prob)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for sequences, labels, seq_lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences, seq_lengths)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}")

# Evaluate on validation set
model.eval()
val_loss = 0.0
with torch.no_grad():
    for sequences, labels, seq_lengths in val_loader:
        outputs = model(sequences, seq_lengths)
        loss = criterion(outputs.squeeze(), labels)
        val_loss += loss.item()
print(f"Validation Loss: {val_loss / len(val_loader):.4f}")




def evaluate_probabilistic_baseline(dataloader):
    """
    Evaluate a probabilistically efficient random baseline on the given dataset.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.

    Returns:
        float: Mean Absolute Error (MAE) of the probabilistic random baseline.
    """
    total_error = 0
    num_samples = 0
    
    # Iterate over the dataloader
    for sequences, labels, seq_lengths in dataloader:
        # Move data to CPU if using CUDA
        seq_lengths = seq_lengths.cpu().numpy()  # Convert to numpy for easier processing
        labels = labels.cpu().numpy()

        batch_size = len(seq_lengths)

        for i in range(batch_size):
            seq_len = seq_lengths[i]
            label = labels[i]

            # Generate a prediction based on the binomial distribution (probabilistic model)
            probabilistic_prediction = np.random.binomial(seq_len, 0.5)

            # Compute absolute error
            error = abs(probabilistic_prediction - label)
            total_error += error
            num_samples += 1

    # Compute Mean Absolute Error (MAE)
    mae = total_error / num_samples
    return mae


probabilistic_baseline_mae = evaluate_probabilistic_baseline(val_loader)
print(f"Probabilistic Random Baseline MAE: {probabilistic_baseline_mae:.4f}")


def generate_binary_sequences(num_samples, min_value=1, max_seq_len=16):
    """
    Generate binary sequences of varying lengths and corresponding labels.
    
    Args:
        num_samples (int): Number of samples to generate.
        min_value (int): The minimum number of 1s in the sequence.
        max_seq_len (int): Maximum length of the sequences.
        
    Returns:
        sequences (list): List of binary sequences.
        labels (list): List of labels (number of 1s in each sequence).
    """
    sequences = []
    labels = []
    
    for _ in range(num_samples):
        # Randomly decide sequence length from 1 to max_seq_len
        seq_len = np.random.randint(1, max_seq_len + 1)
        
        # Create a binary sequence with a random number of 1s between min_value and seq_len
        num_ones = np.random.randint(min_value, seq_len + 1)
        sequence = [1] * num_ones + [0] * (seq_len - num_ones)
        
        # Shuffle the sequence
        np.random.shuffle(sequence)
        
        # Store the sequence and its label (the number of 1s)
        sequences.append(sequence)
        labels.append(num_ones)
    
    return sequences, labels

# Generate evaluation data for sequence lengths from 1 to 32
evaluation_sequences, evaluation_labels = generate_binary_sequences(1000, min_value=1, max_seq_len=32)

# Create a DataLoader for the evaluation dataset
evaluation_dataset = BinarySequenceDataset(evaluation_sequences, evaluation_labels, max_seq_len=32)
evaluation_loader = DataLoader(evaluation_dataset, batch_size=32, shuffle=False)

# Evaluate the trained model on this data
model.eval()
mae_list = []
for sequences, labels, seq_lengths in evaluation_loader:
    sequences, labels = sequences, labels
    
    # Model prediction
    outputs = model(sequences,seq_lengths)
    mae = torch.mean(torch.abs(outputs - labels))
    
    mae_list.append(mae.item())

# Plotting the MAE
import matplotlib.pyplot as plt

# Assuming `mae_list` corresponds to MAE values for sequence lengths 1 to 32
plt.plot(range(1, 33), mae_list, marker='o')
plt.xlabel('Sequence Length')
plt.ylabel('MAE')
plt.title('Model Generalization to Out-of-Distribution Data')
plt.show()