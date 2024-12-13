from sklearn.feature_extraction import FeatureHasher
from models.base_model import SimilaritySearchModel

class HashingSimilarity(SimilaritySearchModel):
    def build(self):
        self.hasher = FeatureHasher(n_features=128, input_type='string')

    def encode(self, image_paths):
        return self.hasher.transform(image_paths).toarray()

    def train(self):
        # No training needed for hashing
        pass