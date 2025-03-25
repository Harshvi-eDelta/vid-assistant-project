if __name__ == '__main__':
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    import cv2
    from torchvision import transforms
    from landmark_cnn import LandmarkCNN

    class LandmarkDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None):
            self.landmarks_frame = pd.read_csv(csv_file)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.landmarks_frame)

        def __getitem__(self, idx):
            img_name = os.path.join(self.img_dir, self.landmarks_frame.iloc[idx, 0])
            # Load the image and get its original size
            image = cv2.imread(img_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]  # Get original image height & width

            # Load landmarks
            landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype("float32").reshape(-1, 2)

            # Normalize landmarks (scale to [0,1] range)
            landmarks[:, 0] /= orig_w  # Normalize X-coordinates
            landmarks[:, 1] /= orig_h  # Normalize Y-coordinates

            # Apply transformations to the image
            if self.transform:
                image = self.transform(image)

            # Flatten landmarks to a single array
            landmarks = landmarks.flatten()
            return image, torch.tensor(landmarks, dtype=torch.float32)


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = LandmarkDataset(csv_file="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/landmarks.csv", img_dir="/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg", transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LandmarkCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, landmarks in train_loader:
            images, landmarks = images.to(device), landmarks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "landmark_model.pth")
    print("Model saved successfully!")
