from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# Load the CLIP model and processor from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def extract_clip_features_batch(image_paths, batch_size=32):
    features_list = []

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        for image_path in batch_paths:
            img = Image.open(image_path).convert("RGB")
            batch_images.append(img)

        # Preprocess the batch of images
        inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Extract features using CLIP
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)

        features_list.append(image_features.cpu().numpy())

    return np.vstack(features_list)  # Stack all batch features
