import os
import torch

img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset/original_jpg_copy"
pth_dir = "/Users/edelta076/Desktop/Project_VID_Assistant/new_dataset//t7_fixed"

missing = []
invalid_shape = []
out_of_bounds = []

for img in os.listdir(img_dir):
    if not img.endswith('.jpg'): continue
    base = os.path.splitext(img)[0]
    pth_path = os.path.join(pth_dir, base + ".pth")
    
    if not os.path.exists(pth_path):
        missing.append(base)
    else:
        data = torch.load(pth_path)
        if data.shape[1] != 2:
            invalid_shape.append(base)
        elif data.min() < 0 or data.max() > 1:
            out_of_bounds.append(base)

print("Missing:", missing)
print("Invalid shapes:", invalid_shape)
print("Out-of-bounds:", out_of_bounds)
