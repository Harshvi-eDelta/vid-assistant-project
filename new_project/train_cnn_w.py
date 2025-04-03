import torch
import torch.optim as optim
import torch.nn as nn
from data_preprocessing_w import get_dataloader
from landmark_cnn_w import get_model
from tqdm import tqdm

# Training Configuration
data_path = "/Users/edelta076/Desktop/Project_VID_Assistant/300W-3D 2/HELEN"
batchsize = 32
epochs = 20
best_loss = float("inf")
lrate = 0.001
patience = 5
counter = 0

# Load Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader = get_dataloader(data_path, batch_size=batchsize)
for images, landmarks in dataloader:
    print("Batch Image Shape:", images.shape)
    print("Batch Landmark Shape:", landmarks.shape)
    break


# Load Model
model = get_model().to(device)
optimizer = optim.Adam(model.parameters(), lr=lrate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
criterion = nn.MSELoss()

# Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
    for images, landmarks in progress_bar:
        images, landmarks = images.to(device), landmarks.to(device)
        
        # Zero gradients, forward pass, compute loss
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")

    # Adjust learning rate
    scheduler.step(avg_loss)
    
    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), "best_landmark_cnn_w.pth")  # Save best model
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

# Step 6: Save Final Model
torch.save(model.state_dict(), "landmark_cnn_w.pth")
print("Model saved successfully!")
