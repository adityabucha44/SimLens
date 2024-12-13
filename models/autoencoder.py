
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from models.base_model import SimilaritySearchModel

class AutoencoderSimilarity(SimilaritySearchModel):
    def build(self):
        input_img = Input(shape=(128, 128, 3))

        # Encoder
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        # Decoder
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        self.model = Model(input_img, decoded)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, train_generator, val_generator):
        self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=50,
            steps_per_epoch=len(train_generator),
            validation_steps=len(val_generator))
        self.model.save("autoencoder_model.h5")