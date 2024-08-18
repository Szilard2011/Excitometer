import torch
import torch.nn as nn
import torch.nn.functional as F

class ExcitometerModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ExcitometerModel, self).__init__()
        
        # Extended architecture
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability

        self.fc1 = nn.Linear(128 * 8 * 8, 1024)  # Adjust dimensions based on input size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(128)

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        
        x = x.view(-1, 128)  # Flatten for the fully connected layer
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Example usage
if __name__ == "__main__":
    model = ExcitometerModel(num_classes=10)
    print(model)

    # Example input tensor with batch size 1, 1 channel, and 64x64 spatial dimensions
    example_input = torch.randn(1, 1, 64, 64)
    output = model(example_input)
    print(output)
