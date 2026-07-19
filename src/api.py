import os
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Import our custom modules
from src.face_recognition.model import build_custom_facenet
from src.face_recognition.inference import recognize_faces_multi, get_face_embedding
from src.medical_analysis.model import build_resnet_model

# Global instances (lazily loaded or pre-initialized)
face_net_detector = None
face_embedding_model = None
medical_model = None
known_database = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that manages application startup and shutdown.
    Replaces deprecated on_event('startup') / on_event('shutdown').
    """
    global face_net_detector, face_embedding_model, medical_model

    # Load face detector
    prototxt_path = "deploy.prototxt"
    caffemodel_path = "face_detector.caffemodel"
    if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
        face_net_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        print("✅ Face detector loaded.")
    else:
        print("⚠ Face detector caffemodel files not found locally.")

    # Load FaceNet model
    try:
        face_embedding_model = build_custom_facenet()
        print("✅ Custom FaceNet embedding model loaded.")
    except Exception as e:
        print(f"⚠ Failed to load custom FaceNet model: {e}")

    # Load Medical model
    try:
        medical_model = build_resnet_model()
        print("✅ ResNet50 medical classification model loaded.")
    except Exception as e:
        print(f"⚠ Failed to load medical classification model: {e}")

    yield  # Runs the application

    # Shutdown / Cleanup operations
    print("Shutting down API server...")

# Initialize FastAPI App with Lifespan
app = FastAPI(
    title="Production Face Recognition & Medical Analysis API",
    description="Enterprise-grade REST API utilizing refactored modular packages.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    """
    Returns API health status.
    """
    return {
        "status": "healthy",
        "face_detector_loaded": face_net_detector is not None,
        "face_embedding_loaded": face_embedding_model is not None,
        "medical_model_loaded": medical_model is not None,
        "registered_users_count": len(known_database)
    }

@app.post("/face/register")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    """
    Extracts a face embedding from an uploaded image and registers the user.
    """
    global face_embedding_model, known_database
    if face_embedding_model is None:
        raise HTTPException(status_code=500, detail="FaceNet embedding model is not initialized.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = get_face_embedding(img_rgb, face_embedding_model)
        known_database[name] = embedding

        return {"status": "success", "message": f"Successfully registered face for '{name}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/face/recognize")
async def recognize_faces(file: UploadFile = File(...), confidence: float = 0.5, matching_threshold: float = 0.5):
    """
    Recognizes faces in a group image and runs emotion detection on them.
    """
    global face_embedding_model, face_net_detector, known_database
    if face_embedding_model is None or face_net_detector is None:
        raise HTTPException(status_code=500, detail="Face recognition sub-system is not fully loaded.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        recognitions = recognize_faces_multi(
            img_bgr=img,
            embedding_model=face_embedding_model,
            known_database=known_database,
            face_net=face_net_detector,
            confidence_threshold=confidence,
            matching_threshold=matching_threshold
        )

        return {"status": "success", "faces_detected": len(recognitions), "results": recognitions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@app.post("/medical/predict")
async def predict_xray(file: UploadFile = File(...)):
    """
    Performs diagnostic binary classification (Normal vs Pneumonia) on a chest X-Ray.
    """
    global medical_model
    if medical_model is None:
        raise HTTPException(status_code=500, detail="Medical diagnostic model is not initialized.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        # Preprocess img
        resized = cv2.resize(img, (224, 224)).astype('float32') / 255.0
        inp = np.expand_dims(resized, axis=0)

        predictions = medical_model.predict(inp, verbose=0)[0]
        normal_prob = float(predictions[0])
        pneumonia_prob = float(predictions[1])

        predicted_class = "Normal" if normal_prob > pneumonia_prob else "Pneumonia"
        confidence = max(normal_prob, pneumonia_prob)

        return {
            "status": "success",
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": {
                "Normal": normal_prob,
                "Pneumonia": pneumonia_prob
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic prediction failed: {str(e)}")
