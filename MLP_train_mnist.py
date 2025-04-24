#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ---- Model Definition ----
class MNISTClassifier(nn.Module):
    def __init__(self, latent_dim=10, num_classes=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # No normalization before first conv
            nn.GLU(dim=1),

            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GLU(dim=1),

            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GLU(dim=1),

            nn.InstanceNorm2d(32)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),  # First FC layer (No LayerNorm)
            nn.GLU(),

            nn.LayerNorm(128),
            nn.Linear(128, 256),  # Second FC layer
            nn.GLU(),

            nn.LayerNorm(128),
            nn.Linear(128, latent_dim),  # Third FC layer (Final latent space)
            nn.LayerNorm(latent_dim)
        )

        # Learnable class embeddings (+1 for out-of-distribution/rejection class)
        self.class_embeddings = nn.Parameter(torch.randn(num_classes + 1, latent_dim))

    def forward(self, x):
        latent = self.fc(self.encoder(x))  # Extract latent representation
        logits = torch.matmul(latent, self.class_embeddings.T)  # Dot product classification
        return logits

    def get_latents(self, x):
        """Returns only the latent vectors without classification logits."""
        with torch.no_grad():
            return self.fc(self.encoder(x))

# ---- Training Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

# ---- Data Loading ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root="/work3/s183995/MNIST/data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root="/work3/s183995/MNIST/data", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---- Training Loop ----
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

# ---- Extract Latents from Test Set ----
os.makedirs("./latents", exist_ok=True)

all_latents = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        latents = model.get_latents(images)
        all_latents.append(latents.cpu())
        all_labels.append(labels.cpu())
        all_images.append(images.cpu())

# Concatenate and save
all_latents = torch.cat(all_latents)
all_labels = torch.cat(all_labels)
all_images = torch.cat(all_images)

torch.save(all_latents, "./latents/test_latents.pt")
torch.save(all_labels, "./latents/test_labels.pt")
torch.save(all_images, "./latents/test_images.pt")

print("Latents saved in './latents/' ✅")
#%%
# ---- Test Accuracy Calculation ----
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)  # Get predicted class
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
#%%