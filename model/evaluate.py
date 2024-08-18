import torch
from torch.utils.data import DataLoader
from model.model import ExcitometerModel
from data.dataset import load_dataset, preprocess_data  # Assuming you have these functions

# Configuration
batch_size = 32
num_classes = 10  # Adjust based on your specific use case

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ExcitometerModel(num_classes=num_classes)
model.to(device)

# Load model weights
model.load_state_dict(torch.load('excitometer_model.pth'))
model.eval()  # Set model to evaluation mode

# Load data
def load_data():
    # Load and preprocess dataset
    test_data = load_dataset('test')  # Replace with actual dataset loading

    test_dataset = Dataset(test_data, preprocess_data)  # Define Dataset class
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

test_loader = load_data()

# Evaluation loop
def evaluate():
    running_loss = 0.0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()  # Assuming a classification problem

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total

    return avg_loss, accuracy

# Run evaluation
test_loss, test_accuracy = evaluate()

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
