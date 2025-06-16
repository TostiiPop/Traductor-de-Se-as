from pydantic import BaseModel, Field
from typing import Optional

class Base64ImageRequest(BaseModel):
    """Modelo para recibir imagen en base64"""
    image_base64: str = Field(..., description="Imagen codificada en base64")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
            }
        }

class ImageUrlRequest(BaseModel):
    """Modelo alternativo para URL de imagen"""
    image_url: str = Field(..., description="URL de la imagen")

class BatchPredictionRequest(BaseModel):
    """Modelo para predicciones en lote"""
    images: list[str] = Field(..., description="Lista de imágenes en base64")
    max_images: Optional[int] = Field(10, description="Máximo número de imágenes a procesar")
