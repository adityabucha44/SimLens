import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm  # For progress tracking

# Load models

resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Preprocess image function
def preprocess_image_batch(image_paths, target_size=(128, 128), grayscale=False):
    color_mode = "grayscale" if grayscale else "rgb"
    batch = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=target_size, color_mode=color_mode)
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        batch.append(img)
    return np.array(batch)

# Feature extraction in batches
def extract_resnet_features_batch(image_paths, batch_size=32):
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch = preprocess_image_batch(image_paths[i:i + batch_size], target_size=(224, 224), grayscale=False)
        batch = preprocess_input(batch)
        batch_features = resnet.predict(batch)
        features.extend(batch_features)
    return np.array(features)




