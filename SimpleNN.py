import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 5

# 2. Data Loading & Preparation
# Transforms convert raw PIL images into PyTorch tensors and scale pixels to [0, 1]
transform = transforms.ToTensor()

# Fetch Training Data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Fetch Testing Data
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 3. Define the Neural Network Architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Flatten the 2D 28x28 image into a 1D vector of 784 pixels
        self.flatten = nn.Flatten()
        # Hidden Layer: 784 inputs -> 128 nodes
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        # Output Layer: 128 nodes -> 10 classes (digits 0 through 9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x # Note: No Softmax here. PyTorch handles it in the loss function.

# Instantiate the model
model = SimpleNet()

# 4. Define the Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. The Training Loop
print("Starting Training...")
for epoch in range(epochs):
    model.train() # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        # Step 1: Clear old gradients
        optimizer.zero_grad()
        
        # Step 2: Forward pass (builds the computation graph)
        outputs = model(images)
        
        # Step 3: Compute the loss
        loss = criterion(outputs, labels)
        
        # Step 4: Backward pass (computes gradients via Autograd)
        loss.backward()
        
        # Step 5: Update weights
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss/len(train_loader):.4f}")

# 6. The Evaluation Loop
print("Starting Evaluation...")
model.eval() # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad(): # Disable gradient tracking for speed and memory efficiency
    for images, labels in test_loader:
        outputs = model(images)
        # outputs contain raw logits. We want the index of the highest logit.
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0) # Count total samples
        correct += (predicted == labels).sum().item() # Count correct predictions

print(f"Final Test Accuracy: {(100 * correct / total):.2f}%")