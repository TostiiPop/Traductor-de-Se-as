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
import sys
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificar que TensorFlow se puede importar
try:
    import tensorflow as tf
    logger.info(f"TensorFlow version: {tf.__version__}")
    tf_available = True
except ImportError as e:
    logger.error(f"TensorFlow not available: {e}")
    tf_available = False

from schemas.response_models import PredictionResponse, HealthResponse
from utils.image_processing import process_uploaded_image

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

# Inicializar el predictor solo si TensorFlow está disponible
predictor = None
if tf_available:
    try:
        from models.predictor import GesturePredictor
        predictor = GesturePredictor()
        logger.info("Predictor inicializado correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar el predictor: {e}")
        predictor = None
else:
    logger.warning("TensorFlow no disponible, predictor deshabilitado")

@app.get("/", response_model=HealthResponse)
async def root():
    """Endpoint de salud de la API"""
    return HealthResponse(
        status="healthy" if tf_available else "limited",
        message="API de Traductor de Lenguaje de Señas funcionando" + ("" if tf_available else " (sin TensorFlow)"),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar el estado de la API y los modelos"""
    if not tf_available:
        return HealthResponse(
            status="unhealthy",
            message="Error: TensorFlow no disponible",
            version="1.0.0"
        )
    
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

@app.get("/debug")
async def debug_info():
    """Información de debug del sistema"""
    debug_info = {
        "python_version": sys.version,
        "tensorflow_available": tf_available,
        "working_directory": os.getcwd(),
        "environment_variables": dict(os.environ),
        "sys_path": sys.path
    }
    
    if tf_available:
        import tensorflow as tf
        debug_info["tensorflow_version"] = tf.__version__
    
    return debug_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(file: UploadFile = File(...)):
    """
    Predecir gesto a partir de una imagen
    """
    if not tf_available:
        raise HTTPException(status_code=503, detail="TensorFlow no disponible")
    
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
    """
    if not tf_available:
        raise HTTPException(status_code=503, detail="TensorFlow no disponible")
    
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
    if not tf_available or predictor is None:
        raise HTTPException(status_code=503, detail="Predictor no disponible")
    
    gestures = predictor.get_available_gestures()
    
    return {
        "success": True,
        "gestures": gestures,
        "total_gestures": len(gestures["one_hand"]) + len(gestures["two_hands"])
    }

@app.get("/model_info")
async def get_model_info():
    """Obtener información sobre los modelos cargados"""
    if not tf_available or predictor is None:
        raise HTTPException(status_code=503, detail="Predictor no disponible")
    
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
