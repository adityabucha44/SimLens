import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense, Conv2D, Flatten, MaxPooling2D
from models.base_model import SimilaritySearchModel

class SiameseNetwork(SimilaritySearchModel):
    def build(self):
        def euclidean_distance(vectors):
            x, y = vectors
            return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

        input_shape = (128, 128, 3)

        # Base network for feature extraction
        base_input = Input(shape=input_shape)
        x = Conv2D(64, (3, 3), activation='relu')(base_input)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        base_network = Model(base_input, x)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        feature_a = base_network(input_a)
        feature_b = base_network(input_b)

        distance = Lambda(euclidean_distance)([feature_a, feature_b])
        self.model = Model([input_a, input_b], distance)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_pairs, val_pairs):
        self.model.fit(
            train_pairs,
            validation_data=val_pairs,
            epochs=25,
            steps_per_epoch=len(train_pairs),
            validation_steps=len(val_pairs))
        self.model.save("siamese_model.h5")
