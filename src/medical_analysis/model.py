import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def build_simple_cnn(input_shape=(224, 224, 3), num_classes=2):
    """
    Variant A — Simple CNN from Scratch (Baseline).
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='Simple_CNN_Scratch')

    return model

def build_resnet_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Variant B — Transfer Learning with ResNet50 (Recommended).
    """
    # Load ResNet50 base with pre-trained ImageNet weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Custom head for classification
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    # Final model mapping inputs to outputs
    model = models.Model(inputs=base_model.input, outputs=predictions, name='ResNet50_Transfer')
    return model
