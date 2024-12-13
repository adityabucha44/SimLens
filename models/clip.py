import torch
from transformers import CLIPProcessor, CLIPModel
from models.base_model import SimilaritySearchModel

class CLIPFeatureExtractor(SimilaritySearchModel):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return self.model.get_image_features(**inputs)

    def train(self):
        # CLIP is pre-trained, no training needed
        pass
