from pydantic import BaseModel
from typing import Optional, Dict, List

class HealthResponse(BaseModel):
    """Modelo de respuesta para el endpoint de salud"""
    status: str
    message: str
    version: str

class PredictionResponse(BaseModel):
    """Modelo de respuesta para predicciones"""
    success: bool
    prediction: str
    confidence: float
    num_hands: int
    processing_time: float
    message: str
    error: Optional[str] = None

class GesturesResponse(BaseModel):
    """Modelo de respuesta para gestos disponibles"""
    success: bool
    gestures: Dict[str, List[str]]
    total_gestures: int

class ModelInfoResponse(BaseModel):
    """Modelo de respuesta para información del modelo"""
    success: bool
    model_info: Dict
