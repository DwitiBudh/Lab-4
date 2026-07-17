import os
import cv2
import numpy as np

# Try importing DeepFace for emotion detection
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

def get_face_embedding(image, embedding_model):
    """
    Given a single face image, preprocesses it and extracts the 128-D embedding.
    """
    if len(image.shape) == 2:  # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize to 160x160 as required by our FaceNet model
    resized = cv2.resize(image, (160, 160)).astype('float32')

    # Scale to [0, 1] if not already
    if resized.max() > 1.0:
        resized /= 255.0

    inp = np.expand_dims(resized, axis=0)
    emb = embedding_model.predict(inp, verbose=0)[0]
    return emb

def register_user(name, image_path, embedding_model, known_database):
    """
    Registers a person in the local dictionary database.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path {image_path} does not exist.")
        return known_database

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return known_database

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    embedding = get_face_embedding(img_rgb, embedding_model)
    known_database[name] = embedding
    print(f"Successfully registered user: '{name}'")
    return known_database

def verify_1to1(img1, img2, embedding_model, threshold=0.5):
    """
    1:1 Face Verification between two images.
    Returns: (is_match, cosine_distance)
    """
    emb1 = get_face_embedding(img1, embedding_model)
    emb2 = get_face_embedding(img2, embedding_model)

    # Cosine distance: 1 - Cosine Similarity
    # Since our embeddings are L2 normalized, Cosine Similarity is just the dot product!
    cosine_sim = np.dot(emb1, emb2)
    distance   = 1.0 - cosine_sim

    return distance < threshold, distance

def detect_emotion_deepface(face_crop):
    """
    Analyzes emotion on a cropped face using DeepFace.
    Returns: (happiness_score, dominant_emotion)
    """
    if not DEEPFACE_AVAILABLE:
        return 0.0, "Unknown (DeepFace disabled)"

    try:
        # DeepFace needs a BGR image
        face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        analysis = DeepFace.analyze(face_bgr, actions=['emotion'], enforce_detection=False, silent=True)
        if isinstance(analysis, list):
            analysis = analysis[0]
        happiness = analysis['emotion']['happy']
        dominant  = analysis['dominant_emotion']
        return happiness, dominant
    except Exception as e:
        print(f"DeepFace emotion analysis error: {e}")
        return 0.0, "Error"

def recognize_faces_multi(img_bgr, embedding_model, known_database, face_net, confidence_threshold=0.5, matching_threshold=0.5):
    """
    Detects multiple faces in an image using OpenCV DNN face detector,
    and identifies each against a database of known face embeddings.
    Also retrieves emotion/happiness scores using DeepFace.

    Returns a list of dicts: [ { 'box': (x1,y1,x2,y2), 'name': '...', 'distance': 0.1, 'happy': 95.5 }, ... ]
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run OpenCV DNN face detection
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            # Scale coordinates back to image dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clip bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face_crop = img_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Extract 128-D Embedding
            embedding = get_face_embedding(face_crop, embedding_model)

            # 1:N Classification via Cosine Distance matching
            min_dist = float('inf')
            matched_name = "Unknown"

            for name, db_embedding in known_database.items():
                cosine_sim = np.dot(embedding, db_embedding)
                distance = 1.0 - cosine_sim
                if distance < min_dist:
                    min_dist = distance
                    matched_name = name

            if min_dist > matching_threshold:
                matched_name = "Unknown"

            # Emotion Analysis
            happy_score, dominant_emotion = detect_emotion_deepface(face_crop)

            results.append({
                'box': (x1, y1, x2, y2),
                'name': matched_name,
                'distance': min_dist if matched_name != "Unknown" else None,
                'happy_percentage': happy_score,
                'dominant_emotion': dominant_emotion
            })

    return results
