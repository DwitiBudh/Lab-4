import os
import sys
import urllib.request
import numpy as np

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Informative logs for dependency checks
print("Checking dependencies...")
try:
    import tensorflow as tf
    from tensorflow import keras
    print("  - TensorFlow: Available")
except ImportError:
    print("❌ ERROR: TensorFlow is missing. Please install it using: pip install tensorflow")
    sys.exit(1)

try:
    import cv2
    print("  - OpenCV: Available")
except ImportError:
    print("❌ ERROR: OpenCV is missing. Please install it using: pip install opencv-python")
    sys.exit(1)

try:
    import sklearn
    from sklearn.datasets import fetch_lfw_people
    print("  - Scikit-learn: Available")
except ImportError:
    print("❌ ERROR: Scikit-learn is missing. Please install it using: pip install scikit-learn")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("  - Matplotlib: Available")
except ImportError:
    print("❌ ERROR: Matplotlib is missing. Please install it using: pip install matplotlib")
    sys.exit(1)

try:
    from deepface import DeepFace
    print("  - DeepFace (Optional): Available")
except ImportError:
    print("  - DeepFace (Optional): NOT Available. Emotion detection will be bypassed.")

# Import from our reusable package modules
try:
    from src.face_recognition.model import build_custom_facenet
    from src.face_recognition.dataset import preprocess_dataset
    from src.face_recognition.inference import recognize_faces_multi, get_face_embedding
    print("✅ Reusable package modules loaded successfully.")
except ImportError as e:
    print(f"❌ ERROR: Failed to import reusable modules from 'src/face_recognition/'. Reason: {e}")
    sys.exit(1)

def ensure_detector_files():
    """
    Downloads the OpenCV DNN face detector prototxt and weights if not present.
    """
    prototxt = "deploy.prototxt"
    weights = "face_detector.caffemodel"

    if not os.path.exists(prototxt):
        print("Downloading face detector prototxt config...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        urllib.request.urlretrieve(url, prototxt)
        print("  - Prototxt downloaded.")

    if not os.path.exists(weights):
        print("Downloading face detector caffe model weights...")
        url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        urllib.request.urlretrieve(url, weights)
        print("  - Model weights downloaded.")

    return prototxt, weights

def main():
    print("\n" + "=" * 60)
    print("   RUNNING INTEGRATED FACIAL RECOGNITION PIPELINE   ")
    print("=" * 60)

    # 1. Load the model
    print("\n[Step 1/4] Initializing FaceNet Embedding Model...")
    embedding_model = build_custom_facenet()
    print("  - Embedding model initialized with L2 normalization output.")

    # 2. Download and configure OpenCV DNN face detector files
    print("\n[Step 2/4] Initializing Face Detector...")
    prototxt, weights = ensure_detector_files()
    face_net = cv2.dnn.readNetFromCaffe(prototxt, weights)
    print("  - Caffe face detector loaded.")

    # 3. Load sample dataset and images
    print("\n[Step 3/4] Loading sample dataset (LFW)...")
    lfw = fetch_lfw_people(min_faces_per_person=2, color=True)
    images = lfw.images
    labels = lfw.target
    target_names = lfw.target_names
    print(f"  - Loaded LFW dataset with {len(images)} images.")

    # Select sample individuals to register in our local database
    unique_ids, counts = np.unique(labels, return_counts=True)
    frequent_ids = unique_ids[counts >= 3][:2]

    known_database = {}
    print("\nRegistering known users into database...")
    for idx, pid in enumerate(frequent_ids):
        name = target_names[pid]
        # Find first image for this person
        p_indices = np.where(labels == pid)[0]
        register_img = images[p_indices[0]]

        # Scale image appropriately to [0, 255] for OpenCV conversion
        if register_img.max() <= 1.0:
            register_img_255 = (register_img * 255).astype(np.uint8)
        else:
            register_img_255 = register_img.astype(np.uint8)

        emb = get_face_embedding(cv2.cvtColor(register_img_255, cv2.COLOR_RGB2BGR), embedding_model)
        known_database[name] = emb
        print(f"  - Registered user: '{name}' using image index {p_indices[0]}")

    # Pick a test image (e.g. the second image of the first registered person)
    test_pid = frequent_ids[0]
    test_p_indices = np.where(labels == test_pid)[0]
    test_image = images[test_p_indices[1]] # different image of the same person
    test_name = target_names[test_pid]

    print(f"\n[Step 4/4] Performing face detection & recognition on test image...")
    print(f"  - Expecting to recognize: '{test_name}'")

    if test_image.max() <= 1.0:
        test_img_bgr = cv2.cvtColor((test_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        test_img_bgr = cv2.cvtColor(test_image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Run multi-recognition
    recognitions = recognize_faces_multi(
        img_bgr=test_img_bgr,
        embedding_model=embedding_model,
        known_database=known_database,
        face_net=face_net,
        confidence_threshold=0.3,
        matching_threshold=0.6
    )

    print(f"\nDetection Summary (Found {len(recognitions)} face(s)):")
    print("-" * 60)
    for r in recognitions:
        name = r['name']
        dist = f"{r['distance']:.4f}" if r['distance'] is not None else "N/A"
        happy = f"{r['happy_percentage']:.1f}%"
        dom = r['dominant_emotion']
        print(f"  - Name: {name:<25} | Distance: {dist:<8} | Happiness: {happy:<6} | Emotion: {dom}")
    print("-" * 60)

    # Plotting the visual output
    print("\nDisplaying visual output (saving plots to 'face_recognition_output.png')...")
    test_img_rgb = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)
    result_img = test_img_rgb.copy()

    for r in recognitions:
        x1, y1, x2, y2 = r['box']
        name = r['name']
        dist = f"({r['distance']:.2f})" if r['distance'] is not None else ""
        color = (50, 200, 50) if name != "Unknown" else (220, 50, 50)

        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_img, f"{name} {dist}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(test_img_rgb)
    axes[0].set_title("Input Sample Image")
    axes[0].axis('off')

    axes[1].imshow(result_img)
    axes[1].set_title("Detection & Recognition Output")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('face_recognition_output.png')
    print("  - Visual output plot saved successfully.")
    print("\n🎉 Pipeline executed successfully!")

if __name__ == "__main__":
    main()
