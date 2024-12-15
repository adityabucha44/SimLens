from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load MobileNetV2 model for feature extraction
mobilenet_v2 = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_mobilenet_features_batch(image_paths, batch_size=32, target_size=(224, 224)):
    features_list = []

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []

        for image_path in batch_paths:
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            batch_images.append(expanded_img_array)

        batch_images = np.vstack(batch_images)
        preprocessed_images = preprocess_input(batch_images)

        # Extract features for the batch
        batch_features = mobilenet_v2.predict(preprocessed_images)
        features_list.append(batch_features)

    return np.vstack(features_list)  # Stack all batch features
