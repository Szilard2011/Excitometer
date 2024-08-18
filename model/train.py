import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model.model import ExcitometerModel
from data.dataset import load_dataset, preprocess_data  # Assuming you have these functions
from data.preprocess import preprocess_data

# Configuration
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 10  # Adjust based on your specific use case

# Initialize the model
model = ExcitometerModel(num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()  # Assuming a classification problem
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load data
def load_data():
    # Load and preprocess dataset
    train_data = load_dataset('train')  # Replace with actual dataset loading
    val_data = load_dataset('val')      # Replace with actual dataset loading

    train_dataset = Dataset(train_data, preprocess_data)  # Define Dataset class
    val_dataset = Dataset(val_data, preprocess_data)      # Define Dataset class

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

train_loader, val_loader = load_data()

# Training loop
def train_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

# Validation loop
def validate_epoch():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

# Training and validation
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = validate_epoch()

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

# Save the model
torch.save(model.state_dict(), 'excitometer_model.pth')

print('Training complete. Model saved as excitometer_model.pth')
