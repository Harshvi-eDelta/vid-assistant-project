import torch
import torch.optim as optim
import torch.nn as nn
from model import HRNet_LandmarkDetector
from data import get_dataloader  # Ensure correct import
from tqdm import tqdm  # ✅ Progress bar

# Hyperparameters
num_epochs = 20
batch_size = 16
learning_rate = 1e-4
data_path = "/Users/edelta076/Desktop/Project_VID_Assistant/300W-3D 2/HELEN"

if __name__ == '__main__':
    # Load Data
    train_loader = get_dataloader(data_path, batch_size=batch_size)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNet_LandmarkDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Loss on heatmaps

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # ✅ Add tqdm progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, landmarks in progress_bar:
            images, landmarks = images.to(device), landmarks.to(device)

            optimizer.zero_grad()
            heatmaps = model(images)  # Output: [batch, 68, 64, 64]

            loss = criterion(heatmaps, landmarks)  # Loss on predicted vs GT heatmaps
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ✅ Update tqdm with current loss
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

    # Save Model
    torch.save(model.state_dict(), "hrnet_landmarks.pth")
    print("Model saved!")
