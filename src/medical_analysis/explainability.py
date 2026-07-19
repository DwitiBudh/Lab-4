import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def build_functional_resnet(input_shape=(224, 224, 3), num_classes=2):
    """
    Rebuilds the ResNet50 classification model using Keras Functional API.
    Exposes base layers directly within the model graph by using input_tensor=inputs.
    This is mandatory for Grad-CAM layer gradient computation.
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, models, Input

    inputs = Input(shape=input_shape, name='medical_input')
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet50_Functional')
    return model, base_model

def generate_gradcam(model, img_array, last_conv_layer_name='conv5_block3_out', pred_index=None):
    """
    Generates Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap.
    """
    return_preds = (pred_index is None)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[0][pred_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        print("⚠ Gradients are None!")
        dummy_heatmap = np.zeros(conv_outputs.shape[1:3])
        if return_preds:
            return dummy_heatmap, predictions.numpy()[0]
        return dummy_heatmap

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    if return_preds:
        return heatmap.numpy(), predictions.numpy()[0]
    else:
        return heatmap.numpy()

def plot_gradcam(original_img, heatmap, intensity=0.4, title="Grad-CAM Activation"):
    """
    Superimposes the Grad-CAM heatmap overlay onto the original X-ray image.
    """
    # Resize heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Scale heatmap to [0, 255]
    heatmap_scaled = np.uint8(255 * heatmap_resized)

    # Apply JET color map to make it a heatmap
    heatmap_color = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)

    # Overlay heatmap onto original image
    if original_img.max() <= 1.0:
        img_uint8 = np.uint8(255 * original_img)
    else:
        img_uint8 = np.uint8(original_img)

    superimposed_img = heatmap_color * intensity + img_uint8
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Convert BGR back to RGB for matplotlib plotting
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_uint8)
    axes[0].set_title("Original Chest X-Ray")
    axes[0].axis('off')

    axes[1].imshow(superimposed_img_rgb)
    axes[1].set_title(title)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    return superimposed_img_rgb
