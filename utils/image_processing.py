import cv2
import numpy as np
from PIL import Image
import io
from typing import Union

def process_uploaded_image(image_data: bytes) -> np.ndarray:
    """
    Procesar imagen subida y convertirla al formato requerido
    
    Args:
        image_data: Datos de la imagen en bytes
        
    Returns:
        Imagen en formato numpy array (BGR)
    """
    try:
        # Convertir bytes a imagen PIL
        image_pil = Image.open(io.BytesIO(image_data))
        
        # Convertir a RGB si es necesario
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Convertir PIL a numpy array
        image_rgb = np.array(image_pil)
        
        # Convertir RGB a BGR (formato de OpenCV)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_bgr
        
    except Exception as e:
        raise ValueError(f"Error al procesar la imagen: {e}")

def resize_image(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """
    Redimensionar imagen manteniendo la proporción
    
    Args:
        image: Imagen en formato numpy array
        max_width: Ancho máximo
        max_height: Alto máximo
        
    Returns:
        Imagen redimensionada
    """
    height, width = image.shape[:2]
    
    # Calcular factor de escala
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # No agrandar la imagen
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

def validate_image(image: np.ndarray) -> bool:
    """
    Validar que la imagen sea válida para procesamiento
    
    Args:
        image: Imagen en formato numpy array
        
    Returns:
        True si la imagen es válida
    """
    if image is None:
        return False
    
    if len(image.shape) != 3:
        return False
    
    height, width, channels = image.shape
    
    if channels != 3:
        return False
    
    if height < 100 or width < 100:
        return False
    
    if height > 4000 or width > 4000:
        return False
    
    return True
