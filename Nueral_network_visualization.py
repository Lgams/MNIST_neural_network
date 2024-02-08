import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import random

# Define the Net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 28*28 is the size of the image, 128 is the output size
        self.fc2 = nn.Linear(128, 64)       # 128 inputs, 64 outputs
        self.fc3 = nn.Linear(64, 10)        # 64 inputs, 10 outputs (one for each digit)

    def forward(self, x):
        x = x.view(-1, 28 * 28)             # Flatten image input
        x = F.relu(self.fc1(x))             # Activation function is relu
        x = F.relu(self.fc2(x))             # Activation function is relu
        x = self.fc3(x)                     # Output so no activation function is needed
        return x

# Instantiate the Net
net = Net()

# Load the trained model parameters
model_path = r'C:\Users\.... *insert your path here*'
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Add map_location if you're on a CPU
net.eval()

# Load the test dataset
mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Choose a few random test images
indices = random.sample(range(len(mnist_testset)), 48)

# Set up the figure
fig, axes = plt.subplots(nrows=8, ncols=6, figsize=(15, 12))

# Disable gradients for evaluation
with torch.no_grad():
    for i, idx in enumerate(indices):
        image, true_label = mnist_testset[idx]
        ax = axes[i//6, i%6]

        # Get the model prediction
        output = net(image.unsqueeze(0))  # Add batch dimension
        _, predicted_label = torch.max(output, 1)

        # Display the image and the prediction
        ax.imshow(image.squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Predicted: {predicted_label.item()}, Truth: {true_label}')

plt.tight_layout()
plt.show()  # This will display the figure with the images and predictions