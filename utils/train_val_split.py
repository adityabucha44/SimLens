import os
import tarfile
import shutil
from sklearn.model_selection import train_test_split
import requests
import zipfile

# Step 1: Download and Extract the Dataset
def download_and_extract(dataset_url, extract_to):
    zip_path = './datasets/caltech101.zip'
    # Download the dataset
    print("Downloading dataset...")
    response = requests.get(dataset_url, stream=True)
    with open(zip_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print("Download complete.")

    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Zip extraction complete.")

    # Check for tar file inside extracted folder
    tar_path = os.path.join(extract_to, "caltech-101", "101_ObjectCategories.tar.gz")
    if os.path.exists(tar_path):
        print("Extracting .tar file...")
        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
        print(".tar extraction complete.")

def split_data(source_dir, train_dir, val_dir, test_dir, val_size=0.2, test_size=0.1):
    """
    Splits the dataset into train, validation, and test sets.

    Args:
    - source_dir: Directory containing the dataset with categories as subdirectories.
    - train_dir: Directory to save training data.
    - val_dir: Directory to save validation data.
    - test_dir: Directory to save test data.
    - val_size: Proportion of the dataset to use for validation (relative to train + val).
    - test_size: Proportion of the dataset to use for testing (relative to the total data).
    """
    print("Splitting dataset into train, validation, and test sets...")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)

            # Split into train+val and test
            train_val_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

            # Further split train+val into train and val
            train_images, val_images = train_test_split(train_val_images, test_size=val_size / (1 - test_size), random_state=42)

            # Create category subdirectories
            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(val_dir, category), exist_ok=True)
            os.makedirs(os.path.join(test_dir, category), exist_ok=True)

            # Copy images to respective directories
            for img in train_images:
                shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))
            for img in val_images:
                shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category, img))
            for img in test_images:
                shutil.copy(os.path.join(category_path, img), os.path.join(test_dir, category, img))

    print("Dataset split completed.")


# Step 3: Run the Complete Workflow
dataset_url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
base_dir = "./datasets"
extract_to = os.path.join(base_dir)
source_dir = os.path.join(base_dir, "101_ObjectCategories")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir=os.path.join(base_dir, "test")

# Download, Extract, and Split
download_and_extract(dataset_url, base_dir)

if os.path.exists(source_dir):
    split_data(source_dir, train_dir, val_dir,test_dir)
else:
    print(f"Source directory not found: {source_dir}")

print("Train directory:", train_dir)
print("Validation directory:", val_dir)