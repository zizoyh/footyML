import os
import shutil
import random

# Input and output paths
input_dir = "data/images/all_frames"
train_dir = "data/ball_dataset/images/train"
val_dir = "data/ball_dataset/images/val"

# Create destination folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Collect all image files
all_images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

# 80/20 split
split_index = int(0.8 * len(all_images))
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Move files
for fname in train_images:
    shutil.copy(os.path.join(input_dir, fname), os.path.join(train_dir, fname))

for fname in val_images:
    shutil.copy(os.path.join(input_dir, fname), os.path.join(val_dir, fname))

print(f"Split {len(all_images)} images into:")
print(f"  {len(train_images)} training images")
print(f"  {len(val_images)} validation images")
