from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam
from scipy.spatial.distance import cdist




def load_and_preprocess_custom_data(data_dir, target_size=(128, 128)):
    """
    Load and preprocess images from a directory structure.
    Args:
    - data_dir: Directory containing labeled subfolders.
    - target_size: Tuple specifying the target size for resizing images.

    Returns:
    - images: Preprocessed image data as a NumPy array.
    - labels: Corresponding labels for the images.
    """
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                # Load, resize, and normalize image
                img = load_img(img_path, target_size=target_size, color_mode="grayscale")
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels



def build_autoencoder():
    """Build and compile the CNN-based autoencoder."""
    # Input layer: accepts images of shape 28x28x1 (MNIST images)
    input_img = Input(shape=(128, 128, 1))

    # Encoder
    # Convolutional layer with 32 filters, each 3x3, using 'relu' activation. 'same' padding ensures output size matches input size.
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    # Max pooling layer to reduce spatial dimensions by half, improving computational efficiency and helping encode positional information.
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Another convolutional layer with 16 filters to further extract features from the image.
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # Reducing spatial dimensions again to further compress the representation.
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Final convolutional layer in the encoder with 8 filters, focusing on the most abstract features of the image.
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # Last max pooling layer in the encoder to achieve the final compressed representation.
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    # Convolutional layer with 8 filters, starting the process of decoding the compressed representation.
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    # Upsampling layer to start expanding the spatial dimensions back to the original size.
    x = UpSampling2D((2, 2))(x)
    # Convolutional layer with 16 filters to further refine the decoded features.
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    # Upsampling again to get closer to the original image size.
    x = UpSampling2D((2, 2))(x)
    # Convolutional layer with 32 filters, nearly restoring the original depth of features.
    # x = Conv2D(32, (3, 3), activation='relu')(x)  # Note: No padding here, changes size slightly.
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # Fixed here

    # Final upsampling to match the original image dimensions.
    x = UpSampling2D((2, 2))(x)
    # Output layer to reconstruct the original image. Uses 'sigmoid' activation to output pixel values between 0 and 1.
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Compiling the autoencoder model with Adam optimizer and binary cross-entropy loss.
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
    return autoencoder


def train_autoencoder(autoencoder, x_train, x_test):
    """Train the autoencoder."""
    autoencoder.fit(x_train, x_train, epochs=1, batch_size=64, shuffle=True, validation_data=(x_test, x_test))

def generate_embeddings(encoder, x_test):
    """Generate embeddings for the test set."""
    return encoder.predict(x_test)

def find_similar_images(embeddings, selected_indices):
    """Find and return indices of similar images based on embeddings."""
    similar_images_indices = []
    for index in selected_indices:
        distances = cdist(embeddings[index:index+1], embeddings, 'euclidean')
        closest_indices = np.argsort(distances)[0][1:4]  # Exclude self
        similar_images_indices.append(closest_indices)
    return similar_images_indices

def display_similar_images(x_test, selected_indices, similar_images_indices):
    """Visualize the original and similar images."""
    plt.figure(figsize=(10, 7))
    for i, (index, sim_indices) in enumerate(zip(selected_indices, similar_images_indices)):
        ax = plt.subplot(3, 4, i * 4 + 1)
        plt.imshow(x_test[index].reshape(128, 128))
        plt.title("Original")
        plt.gray()
        ax.axis('off')

        for j, sim_index in enumerate(sim_indices):
            ax = plt.subplot(3, 4, i * 4 + j + 2)
            plt.imshow(x_test[sim_index].reshape(128, 128))
            plt.title(f"Similar {j+1}")
            plt.gray()
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# Main workflow

# Load data
train_dir,val_dir="datasets/train","datasets/val"
x_train,_ = load_and_preprocess_custom_data(train_dir)   
x_test,_ = load_and_preprocess_custom_data(val_dir)


autoencoder = build_autoencoder()
train_autoencoder(autoencoder, x_train, x_test)

encoder = Model(autoencoder.input, autoencoder.layers[-7].output)
encoded_imgs = generate_embeddings(encoder, np.reshape(x_test, (len(x_test), 128, 128, 1)))
encoded_imgs_flatten = encoded_imgs.reshape((len(x_test), np.prod(encoded_imgs.shape[1:])))

np.random.seed(0)
selected_indices = np.random.choice(x_test.shape[0], 3, replace=False)
similar_images_indices = find_similar_images(encoded_imgs_flatten, selected_indices)
display_similar_images(x_test, selected_indices, similar_images_indices)
