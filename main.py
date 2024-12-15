import os
from utils import data_generator
from utils import train_val_split
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from methods import autoencoders,clip,vgg16,resnet,vit,mobilenet
from evaluate.evaluate import evaluate_model_top_k_binary



dataset_url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
base_dir = "./datasets"
extract_to = os.path.join(base_dir)
source_dir = os.path.join(base_dir, "101_ObjectCategories")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir=os.path.join(base_dir, "test")

# Download, Extract, and Split
train_val_split.download_and_extract(dataset_url, base_dir)

if os.path.exists(source_dir):
    train_val_split.split_data(source_dir, train_dir, val_dir,test_dir)
else:
    print(f"Source directory not found: {source_dir}")

print("Train directory:", train_dir)
print("Validation directory:", val_dir)



# Dataset directory
dataset_dir = "./datasets/test"
image_paths = []
labels = []

# Collect image paths and labels
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.endswith((".jpg", ".png")):
                image_paths.append(os.path.join(folder_path, img_file))
                labels.append(folder)  # Folder name as the label


autoencoders_features=autoencoders.extract_autoencoder_features_batch(image_paths,batch_size=32)
resnet_features=resnet.extract_resnet_features_batch(image_paths,batch_size=32)
vgg16_features = vgg16.extract_vgg16_features_batch(image_paths, batch_size=32)
mobilenet_features = mobilenet.extract_mobilenet_features_batch(image_paths, batch_size=32)
vit_features = vit.extract_vit_features_batch(image_paths, batch_size=32)
clip_features = clip.extract_clip_features_batch(image_paths, batch_size=32)


# # Print the shapes of the extracted features

# print("Autoencoder features shape:", autoencoders_features.shape)  # Should be (n_samples, n_features)
# print("ResNet features shape:", resnet_features.shape)  # Should be (n_samples, n_features)
# print("VGG16 features shape:", vgg16_features.shape)  # Should be (n_samples, n_features)
# print("MobileNet features shape:", mobilenet_features.shape)  # Should be (n_samples, n_features)
# print("ViT features shape:", vit_features.shape)  # Should be (n_samples, n_features)
# print("CLIP features shape:", clip_features.shape)  # Should be (n_samples, n_features)





# Evaluate models
autoencoder_results_top_k_binary = evaluate_model_top_k_binary(autoencoders_features, labels, model_name="Autoencoder", k=5)
resnet_results_top_k_binary = evaluate_model_top_k_binary(resnet_features, labels, model_name="ResNet", k=5)
vgg16_results_top_k_binary = evaluate_model_top_k_binary(vgg16_features, labels, model_name="vgg16", k=5)
mobilenet_results_top_k_binary = evaluate_model_top_k_binary(mobilenet_features, labels, model_name="MobileNet", k=5)
Vit_results_top_k_binary = evaluate_model_top_k_binary(vit_features, labels, model_name="ViT", k=5)
clip_results_top_k_binary = evaluate_model_top_k_binary(clip_features, labels, model_name="Clip", k=5)

