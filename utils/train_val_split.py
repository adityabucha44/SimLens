import os
import shutil
from sklearn.model_selection import train_test_split

source_dir = 'datasets/caltech-101/101_ObjectCategories'
train_dir = 'datasets/caltech-101/train'
val_dir = 'datasets/caltech-101/val'

# Create train and val directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Iterate over each category folder
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)

    if os.path.isdir(category_path):
        # List all images in the category
        images = os.listdir(category_path)
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

        # Create train/val category folders
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)

        # Move images to respective folders
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))
        for img in val_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category, img))

print("Train and Validation splits created successfully!")
