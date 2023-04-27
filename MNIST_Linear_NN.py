import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tf
import matplotlib.pyplot as plt

print("GPU available" if torch.cuda.is_available() else "GPU not available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28*28
hidden_layer_1_size = 300
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Importing MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = tf.ToTensor(), download = False)
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = tf.ToTensor(), download = False)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

examples = iter(train_loader)
samples, labels = examples.next()
# samples = 100x1x28x28, labels = 100x1

# normalization of training data set
mean_px = samples.mean()
std_px = samples.std()
samples = (samples - mean_px)/(std_px)

"""
# display 9 random images from the MNIST dataset
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
plt.show()
"""

# defining the NeuralNet class
class NeuralNet(nn.Module):
    # definition of layers
    def __init__(self, input_size, hidden_layer_1_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_1_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer_1_size, num_classes)

    # operations on a forward pass
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

# creating an object of the NeuralNet class
model = NeuralNet(input_size, hidden_layer_1_size, num_classes).to(device)

# loss and optimizer
cross_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
n_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100x1x28x28 converted to 100x784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = cross_loss(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%200==0:
            print(f'epoch {epoch+1}/{num_epochs} - step {i+1}/{n_steps}, loss = {loss.item():.4f}')

# test loop (no backprop)
with torch.no_grad():
    n_correct=0
    n_samples=0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    accuracy = n_correct/n_samples * 100.0
    print(f'accuracy = {accuracy}')