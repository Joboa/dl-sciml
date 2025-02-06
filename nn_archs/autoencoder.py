import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


batch_size = 64
transform = transforms.ToTensor()

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])

mnist_data = datasets.MNIST(root="./data",
                            train=True,
                            download=True,
                            transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=batch_size,
                                          shuffle=True)

# Access the data
data_iter = iter(data_loader)
images, labels = next(data_iter) 
print(images.shape)

# Plot one image
# plt.imshow(images[0].reshape(28, 28), cmap='gray')
# plt.show()

# Plot mulitple images
# fig, axes = plt.subplots(1, 9, figsize=(8,8))
# for i in range(9):
#     axes[i].imshow(images[i].reshape(28,28), cmap="gray")
#     axes[i].axis("off")
# plt.show()


# print(torch.min(images), torch.max(images)) 
# # the range is needed in selection of activation function

class AutoencoderANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128), # image size = 28 x 28
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12), # image size = 28 x 28
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Actual image: N, 1, 28, 28
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # N, 16, 14, 14
            nn.ReLU(),
            # nn.MaxPool2d()
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7) # N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            # starting point: N, 64, 1, 1
            nn.ConvTranspose2d(64, 32, 7), # N, 32, 7, 7
            nn.ReLU(),
            # nn.MaxUnpool1d()
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AutoencoderCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Training
n_epochs = 10
outputs = []
for epoch in range(n_epochs):
    for (img, _) in data_loader:

        optimizer.zero_grad()

        # img = img.reshape(-1, 28*28) # AutoencoderANN
        # For AutoencoderANN, image_size = (1, 28 * 28)
        # For AutoencoderCNN, image_size = (1, 28, 28)
        recon_img = model(img)
        loss = criterion(recon_img, img)

        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
    outputs.append((epoch, img, recon_img))

# Plot images
for k in range(0, n_epochs, 4):
    plt.figure(figsize=(9,2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon_imgs = outputs[k][2].detach().numpy()

    for i, item in enumerate(imgs):
        if i >=9: break
        plt.subplot(2, 9, i+1)
        item = item.reshape(-1, 28,28)
        plt.imshow(item[0])

    for i, item in enumerate(recon_imgs):
        if i >=9: break
        plt.subplot(2, 9, 9+i+1) # start on second row
        item = item.reshape(-1, 28,28) 
        plt.imshow(item[0])
    plt.savefig(f"{k}_image.png")
    plt.show()