from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import VGG16
from models.base_model import SimilaritySearchModel

class PretrainedCNNFeatureExtractor(SimilaritySearchModel):
    def build(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        output = Dense(128, activation='relu')(x)
        self.model = Model(base_model.input, output)
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, train_generator, val_generator):
        self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=20,
            steps_per_epoch=len(train_generator),
            validation_steps=len(val_generator))
        self.model.save("pretrained_cnn_model.h5")
