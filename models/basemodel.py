class SimilaritySearchModel:
    def __init__(self):
        self.model = None

    def build(self):
        raise NotImplementedError

    def train(self, train_generator, val_generator):
        raise NotImplementedError