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

from models.predictor import GesturePredictor
from schemas.response_models import PredictionResponse, HealthResponse
from utils.image_processing import process_uploaded_image

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear instancia de FastAPI
app = FastAPI(
    title="Traductor de Lenguaje de Señas API",
    description="API para reconocimiento de gestos estáticos en lenguaje de señas",
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

# Inicializar el predictor
try:
    predictor = GesturePredictor()
    logger.info("Predictor inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar el predictor: {e}")
    predictor = None

@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint de salud de la API"""
    return HealthResponse(
        status="healthy",
        message="API de Traductor de Lenguaje de Señas funcionando correctamente",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar el estado de la API y los modelos"""
    if predictor is None:
        return HealthResponse(
            status="unhealthy",
            message="Error: Predictor no inicializado",
            version="1.0.0"
        )
    
    model_status = predictor.check_models_status()
    
    return HealthResponse(
        status="healthy" if model_status["all_loaded"] else "partial",
        message=f"Modelos cargados: {model_status}",
        version="1.0.0"
    )

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
        raise HTTPException(status_code=500, detail="Predictor no disponible")
    
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
async def predict_gesture_base64(image_base64: str):
    """
    Predecir gesto a partir de una imagen en base64
    
    Args:
        image_base64: Imagen codificada en base64
    
    Returns:
        Predicción del gesto con confianza
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor no disponible")
    
    try:
        # Decodificar imagen base64
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
        raise HTTPException(status_code=500, detail="Predictor no disponible")
    
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
        raise HTTPException(status_code=500, detail="Predictor no disponible")
    
    info = predictor.get_model_info()
    
    return {
        "success": True,
        "model_info": info
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
