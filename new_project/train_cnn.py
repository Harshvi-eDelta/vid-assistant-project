import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Initialize model, optimizer, and loss function
model = LandmarkCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
criterion = nn.SmoothL1Loss()

# Create DataLoader for batch processing
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training Loop
num_epochs = 15
best_loss = float("inf")
patience = 5
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, landmarks in progress_bar:
        images, landmarks = images.to(device), landmarks.to(device)
        
        # Zero gradients, perform forward pass, compute loss
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    # Adjust learning rate based on loss
    scheduler.step(avg_loss)
    
    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), "best_landmark_model.pth")  # Save best model
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break

# Save the final model
torch.save(model.state_dict(), "landmark_model.pth")
print("Model saved successfully!")
