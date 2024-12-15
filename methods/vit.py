import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image

# Load ViT model and feature extractor
device = "cuda" if torch.cuda.is_available() else "cpu"
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def extract_vit_features_batch(image_paths, batch_size=32, target_size=(224, 224)):
    features_list = []

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        for image_path in batch_paths:
            img = Image.open(image_path).convert("RGB").resize(target_size)
            batch_images.append(img)

        # Preprocess the batch of images
        inputs = vit_feature_extractor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract features using ViT
        with torch.no_grad():
            outputs = vit_model(**inputs)

        batch_features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        features_list.append(batch_features)

    return np.vstack(features_list)  # Stack all batch features
