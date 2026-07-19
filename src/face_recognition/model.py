import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_custom_facenet(input_shape=(160, 160, 3), embedding_size=128):
    """
    Lightweight FaceNet-inspired CNN.
    Outputs L2-normalized 128-D embeddings.
    """
    inputs = keras.Input(shape=input_shape, name='face_input')
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(embedding_size)(x)

    # L2 Normalization Layer
    outputs = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name='l2_normalization'
    )(x)

    return keras.Model(inputs, outputs, name='FaceNet_Embedding')

def triplet_loss(margin=0.5):
    """
    Custom Keras Triplet Loss.
    y_pred shape: (batch, 384) = concat of [anchor(128), positive(128), negative(128)]
    """
    def loss(y_true, y_pred):
        # Slice the concatenated prediction vector back into its components
        anchor   = y_pred[:, 0:128]
        positive = y_pred[:, 128:256]
        negative = y_pred[:, 256:384]

        # Calculate distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        # Loss formula: max(pos_dist - neg_dist + margin, 0)
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    return loss

def build_siamese_network(embedding_model, input_shape=(160, 160, 3)):
    """
    Siamese Network wrapper: 3 inputs (Anchor, Positive, Negative), shared FaceNet weights.
    Concatenates the three 128-D embeddings into a single (batch, 384) output vector.
    """
    input_a = keras.Input(shape=input_shape, name='anchor')
    input_p = keras.Input(shape=input_shape, name='positive')
    input_n = keras.Input(shape=input_shape, name='negative')

    # Shared weights (same embedding model instance)
    emb_a = embedding_model(input_a)
    emb_p = embedding_model(input_p)
    emb_n = embedding_model(input_n)

    # Concatenate the outputs
    merged_output = layers.concatenate([emb_a, emb_p, emb_n], axis=1)

    return keras.Model(inputs=[input_a, input_p, input_n], outputs=merged_output, name='Siamese_Network')
