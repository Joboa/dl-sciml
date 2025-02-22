import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

n_epochs = 4
batch_size = 4
learning_rate = 1e-3

# dataset range [0,1] and normalized range [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

train_dataset = datasets.CIFAR10(root="./data", train=True,
                                 download=True, transform=transform)

test_dataset = datasets.CIFAR10(root="./data", train=False,
                                 download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=True)

# Multi class
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)

# def imshow(img):
#     img = img / 2 + 0.5 
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# # get some random training images
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# imshow(torchvision.utils.make_grid(images))


# Network
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6, 5) # input_channnel_size = 3, # output_channnel_size = 6, kernel = 5
        self.pool = nn.MaxPool2d(2,2) # kernel_size=2, stride=2
        self.conv2 = nn.Conv2d(6,16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 for the multiclass output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training
n__total_steps = len(train_loader)
print("Data length: ", n__total_steps)
for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        # image shape: (4,3,32,32)
        # images = images.to(device)
        # labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'Epoch: {epoch+1}/{n_epochs},, Step: {i+1}/{n__total_steps}, Loss: {loss.item():.4f}')
print("Training completed")

# Model Evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_correct_class = [0] * 10
    n_class_samples = [0] * 10

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_correct_class[label] +=1 
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of model: {acc}%")

    for i in range(10):
        acc = 100.0 * n_correct_class[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]}: {acc}%")