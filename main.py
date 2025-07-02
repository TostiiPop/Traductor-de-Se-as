from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional
import logging
import time

from models.predictor import GesturePredictor
from schemas.response_models import (
    PredictionResponse, 
    HealthResponse, 
    FaceDistanceResponse,
    DistanceThresholdsResponse
)
from utils.image_processing import process_uploaded_image
from utils.face_distance_detector import FaceDistanceDetector
from schemas.request_models import Base64ImageRequest, ImageUrlRequest

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear instancia de FastAPI
app = FastAPI(
    title="Traductor de Lenguaje de Señas API",
    description="API para reconocimiento de gestos estáticos en lenguaje de señas y detección de distancia facial",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir peticiones desde Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el predictor de gestos
try:
    predictor = GesturePredictor()
    logger.info("Predictor de gestos inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar el predictor de gestos: {e}")
    predictor = None

# Inicializar el detector de distancia facial (independiente)
try:
    face_distance_detector = FaceDistanceDetector()
    logger.info("Detector de distancia facial inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar el detector de distancia facial: {e}")
    face_distance_detector = None

@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint de salud de la API"""
    return HealthResponse(
        status="healthy",
        message="API de Traductor de Lenguaje de Señas y Detector de Distancia Facial funcionando correctamente",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar el estado de la API y los modelos"""
    gesture_status = "OK" if predictor is not None else "ERROR"
    distance_status = "OK" if face_distance_detector is not None else "ERROR"
    
    if predictor is None and face_distance_detector is None:
        return HealthResponse(
            status="unhealthy",
            message="Error: Ningún servicio disponible",
            version="1.0.0"
        )
    
    status_message = f"Gestos: {gesture_status}, Distancia: {distance_status}"
    
    if predictor is not None:
        model_status = predictor.check_models_status()
        status_message += f", Modelos: {model_status}"
    
    return HealthResponse(
        status="healthy" if (predictor is not None or face_distance_detector is not None) else "unhealthy",
        message=status_message,
        version="1.0.0"
    )

# ==================== ENDPOINTS DE GESTOS ====================

@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(file: UploadFile = File(...)):
    """
    Predecir gesto a partir de una imagen
    
    Args:
        file: Imagen en formato JPG, PNG, etc.
    
    Returns:
        Predicción del gesto con confianza
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor de gestos no disponible")
    
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer y procesar la imagen
        image_data = await file.read()
        image = process_uploaded_image(image_data)
        
        # Realizar predicción
        result = predictor.predict_from_image(image)
        
        return PredictionResponse(
            success=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            num_hands=result["num_hands"],
            processing_time=result["processing_time"],
            message="Predicción realizada exitosamente"
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

@app.post("/predict_base64", response_model=PredictionResponse)
async def predict_gesture_base64(request: Base64ImageRequest):
    """
    Predecir gesto a partir de una imagen en base64
    
    Args:
        request: Objeto con imagen codificada en base64
    
    Returns:
        Predicción del gesto con confianza
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor de gestos no disponible")
    
    try:
        # Decodificar imagen base64
        image_base64 = request.image_base64
        
        # Remover prefijo si existe (data:image/jpeg;base64,)
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = process_uploaded_image(image_data)
        
        # Realizar predicción
        result = predictor.predict_from_image(image)
        
        return PredictionResponse(
            success=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            num_hands=result["num_hands"],
            processing_time=result["processing_time"],
            message="Predicción realizada exitosamente"
        )
        
    except Exception as e:
        logger.error(f"Error en predicción base64: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

@app.get("/available_gestures")
async def get_available_gestures():
    """Obtener lista de gestos disponibles"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor de gestos no disponible")
    
    gestures = predictor.get_available_gestures()
    
    return {
        "success": True,
        "gestures": gestures,
        "total_gestures": len(gestures["one_hand"]) + len(gestures["two_hands"])
    }

@app.get("/model_info")
async def get_model_info():
    """Obtener información sobre los modelos cargados"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor de gestos no disponible")
    
    info = predictor.get_model_info()
    
    return {
        "success": True,
        "model_info": info
    }

# ==================== ENDPOINTS DE DISTANCIA FACIAL ====================

@app.post("/face_distance", response_model=FaceDistanceResponse)
async def detect_face_distance(file: UploadFile = File(...)):
    """
    Detectar distancia facial a partir de una imagen
    
    Args:
        file: Imagen en formato JPG, PNG, etc.
    
    Returns:
        Información detallada sobre la distancia de la cara
    """
    if face_distance_detector is None:
        raise HTTPException(status_code=500, detail="Detector de distancia facial no disponible")
    
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer y procesar la imagen
        image_data = await file.read()
        image = process_uploaded_image(image_data)
        
        # Detectar distancia facial
        result = face_distance_detector.detect_distance(image)
        
        return FaceDistanceResponse(**result)
        
    except Exception as e:
        logger.error(f"Error en detección de distancia facial: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

@app.post("/face_distance_base64", response_model=FaceDistanceResponse)
async def detect_face_distance_base64(request: Base64ImageRequest):
    """
    Detectar distancia facial a partir de una imagen en base64
    
    Args:
        request: Objeto con imagen codificada en base64
    
    Returns:
        Información detallada sobre la distancia de la cara
    """
    if face_distance_detector is None:
        raise HTTPException(status_code=500, detail="Detector de distancia facial no disponible")
    
    try:
        # Decodificar imagen base64
        image_base64 = request.image_base64
        
        # Remover prefijo si existe (data:image/jpeg;base64,)
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = process_uploaded_image(image_data)
        
        # Detectar distancia facial
        result = face_distance_detector.detect_distance(image)
        
        return FaceDistanceResponse(**result)
        
    except Exception as e:
        logger.error(f"Error en detección de distancia facial base64: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

@app.get("/face_distance_thresholds", response_model=DistanceThresholdsResponse)
async def get_face_distance_thresholds():
    """
    Obtener los umbrales de distancia configurados
    
    Returns:
        Información sobre los umbrales y rangos de distancia
    """
    if face_distance_detector is None:
        raise HTTPException(status_code=500, detail="Detector de distancia facial no disponible")
    
    try:
        thresholds_info = face_distance_detector.get_distance_thresholds()
        
        return DistanceThresholdsResponse(
            success=True,
            **thresholds_info
        )
        
    except Exception as e:
        logger.error(f"Error al obtener umbrales de distancia: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener información: {str(e)}")

@app.post("/check_optimal_distance")
async def check_optimal_distance(request: Base64ImageRequest):
    """
    Verificar si la distancia es óptima (endpoint rápido para validación)
    
    Args:
        request: Objeto con imagen codificada en base64
    
    Returns:
        Respuesta simple indicando si la distancia es óptima
    """
    if face_distance_detector is None:
        raise HTTPException(status_code=500, detail="Detector de distancia facial no disponible")
    
    try:
        # Decodificar imagen base64
        image_base64 = request.image_base64
        
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = process_uploaded_image(image_data)
        
        # Detectar distancia
        result = face_distance_detector.detect_distance(image)
        
        return {
            "success": result["success"],
            "face_detected": result["face_detected"],
            "is_optimal": result["is_optimal"],
            "distance_category": result["distance_category"],
            "quick_recommendation": result["recommendation"] if not result["is_optimal"] else "Mantén esta posición"
        }
        
    except Exception as e:
        logger.error(f"Error en verificación rápida de distancia: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

# ==================== ENDPOINTS DE DEBUG ====================

@app.post("/debug_base64")
async def debug_base64_request(request: dict):
    """Endpoint para debuggear el formato que llega"""
    logger.info(f"Received request keys: {list(request.keys())}")
    logger.info(f"Request type: {type(request)}")
    
    if 'image_base64' in request:
        image_data = request['image_base64']
        logger.info(f"Image data length: {len(image_data)}")
        logger.info(f"Image data starts with: {image_data[:50]}...")
        
        return {
            "success": True,
            "message": "Debug successful",
            "data_length": len(image_data),
            "starts_with": image_data[:50],
            "has_comma": ',' in image_data
        }
    else:
        return {
            "success": False,
            "message": "No image_base64 field found",
            "received_fields": list(request.keys())
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
