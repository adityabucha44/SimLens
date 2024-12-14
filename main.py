import os
from utils.data_utils import create_data_generators
from models.autoencoder import AutoencoderSimilarity
from models.pretrained_cnn import PretrainedCNNFeatureExtractor
from models.siamese import SiameseNetwork
# from models.clip import CLIPFeatureExtractor
from models.hashing import HashingSimilarity

data_dir = "./datasets/caltech-101/"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

def main():
    # Prepare data generators
    train_generator, val_generator = create_data_generators(train_dir, val_dir)

    # Train and evaluate Autoencoder approach
    print("Training Autoencoder approach...")
    autoencoder_model = AutoencoderSimilarity()
    autoencoder_model.build()
    autoencoder_model.train(train_generator, val_generator)

    # Train and evaluate Pre-trained CNN approach
    print("Training Pre-trained CNN approach...")
    cnn_model = PretrainedCNNFeatureExtractor()
    cnn_model.build()
    cnn_model.train(train_generator, val_generator)

    # Placeholder for Siamese Network training
    print("Training Siamese Network approach...")
    siamese_model = SiameseNetwork()
    siamese_model.build()
    # Example: train_siamese_pairs should be prepared
    # siamese_model.train(train_siamese_pairs, val_siamese_pairs)

    # Placeholder for CLIP-based feature extraction
    print("Evaluating CLIP-based feature extraction...")
    # clip_model = CLIPFeatureExtractor()

    # Placeholder for Hashing-based similarity
    print("Using Hashing-based similarity...")
    hashing_model = HashingSimilarity()
    hashing_model.build()

    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()
