import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

def load_lfw_dataset(min_faces_per_person=2, color=True):
    """
    Downloads and loads the Labeled Faces in the Wild (LFW) dataset.
    """
    print("Downloading Labeled Faces in the Wild (LFW)...")
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, color=color)
    return lfw.images, lfw.target, lfw.target_names

def preprocess_dataset(imgs, size=(160, 160)):
    """
    Resizes each image in the array to specified dimensions and normalizes pixel values to [0, 1].
    """
    processed = []
    for img in imgs:
        # Resize to specified size (defaults to 160x160)
        resized = cv2.resize(img, size)

        # Ensure values are float32 scaled to [0, 1]
        if resized.max() > 1.0:
            resized = resized.astype('float32') / 255.0
        else:
            resized = resized.astype('float32')

        processed.append(resized)
    return np.array(processed)

def generate_real_triplets(X, y, num_triplets=500):
    """
    Offline triplet mining:
      - Anchor & Positive = Two different images of the SAME person.
      - Negative          = An image of a DIFFERENT person.
    """
    anchors   = []
    positives = []
    negatives = []

    unique_classes = np.unique(y)

    # Pre-map indices for each class/person for rapid lookup
    class_to_indices = {c: np.where(y == c)[0] for c in unique_classes}

    # Filter classes with at least 2 images (required for Anchor + Positive)
    valid_classes = [c for c, idxs in class_to_indices.items() if len(idxs) >= 2]

    if not valid_classes:
        raise ValueError("No classes have at least 2 samples to generate triplets.")

    for _ in range(num_triplets):
        # 1. Pick a random person (class)
        anchor_class = np.random.choice(valid_classes)
        class_idxs   = class_to_indices[anchor_class]

        # 2. Select two different images of this person (Anchor, Positive)
        a_idx, p_idx = np.random.choice(class_idxs, size=2, replace=False)

        # 3. Pick a different person for the negative image
        neg_class = np.random.choice([c for c in unique_classes if c != anchor_class])
        neg_idxs  = class_to_indices[neg_class]
        n_idx     = np.random.choice(neg_idxs)

        anchors.append(X[a_idx])
        positives.append(X[p_idx])
        negatives.append(X[n_idx])

    return [np.array(anchors), np.array(positives), np.array(negatives)]
