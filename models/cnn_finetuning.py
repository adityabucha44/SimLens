from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from models.basemodel import SimilaritySearchModel

class PretrainedCNNFeatureExtractor(SimilaritySearchModel):
    def build(self):
        # Load Pre-trained ResNet50 and Fine-Tune
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(len(train_generator.class_indices), activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=output)

        # Freeze base model layers for fine-tuning
        for layer in base_model.layers:
            layer.trainable = False

        # Compile Model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    def train(self, train_generator, val_generator):
        # Fit the model without labels (use images themselves as the target)
        self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=20,
            steps_per_epoch=len(train_generator),
            validation_steps=len(val_generator))

        # Save the trained model
        self.model.save("pretrained_cnn_model.h5")