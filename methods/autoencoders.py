import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm  # For progress tracking

# Load models
autoencoder = load_model("autoencoder_model_15_epochs.h5")
encoder = Model(autoencoder.input, autoencoder.layers[-7].output)

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
def extract_autoencoder_features_batch(image_paths, batch_size=32):
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch = preprocess_image_batch(image_paths[i:i + batch_size], target_size=(128, 128), grayscale=True)
        batch_features = encoder.predict(batch)
        batch_flattened = batch_features.reshape(batch_features.shape[0], -1)  # Flatten each feature map
        features.extend(batch_flattened)
    return np.array(features)





