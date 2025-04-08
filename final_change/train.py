import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import LandmarkCNN
from data import LandmarkDataset
from tqdm import tqdm  # ✅ import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # ✅ Resize to a fixed size
    transforms.ToTensor()
])

train_dataset = LandmarkDataset(
    "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy",
    "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/t7",
    transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = LandmarkCNN().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

os.makedirs("checkpoints", exist_ok=True)
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for images, landmarks in progress_bar:
        images, landmarks = images.to(device), landmarks.to(device)
        outputs = model(images)
        loss = criterion(outputs, landmarks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} - Average Loss: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), f"checkpoints/landmark_epoch{epoch}.pth")
