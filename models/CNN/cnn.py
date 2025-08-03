import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskCNN(nn.Module):
    def __init__(self, task='classification'):
        super(MultiTaskCNN, self).__init__()
        self.task = task

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input channels = 1 (grayscale), output channels = 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling

        # Calculate the flattened size after convolutions and pooling
        # With 128x128 input, after 3 pooling layers (each reduces size by half):
        # 128 -> 64 -> 32 -> 16
        flattened_size = 128 * 16 * 16  # Output channels (128) * 16x16 spatial size

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output layer - differs for classification and regression
        if task == 'classification':
            self.output = nn.Linear(64, 10)  # 10 classes (0-9) for digit count classification
        elif task == 'regression':
            self.output = nn.Linear(64, 1)   # Single output for regression

    def forward(self, x):
        # Pass through convolutional layers with activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output from convolutional layers
        x = x.view(-1, 128 * 16 * 16)

        # Fully connected layers with activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer with different activation for classification or regression
        if self.task == 'classification':
            x = self.output(x)
            x = F.log_softmax(x, dim=1)  # Log softmax for classification
        elif self.task == 'regression':
            x = self.output(x)  # Linear output for regression (no activation)

        return x



class MultiLabelCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MultiLabelCNN, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Define pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Adjust the size according to the image size after conv layers
        self.fc2 = nn.Linear(256, num_classes)

        # Define dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Convolutional layers with activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer with sigmoid activation for multi-label classification
        x = torch.sigmoid(self.fc2(x))
        
        return x