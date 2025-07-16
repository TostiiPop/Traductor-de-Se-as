import cv2
import numpy as np
from PIL import Image
import io
from typing import Union

def process_uploaded_image(image_data: bytes) -> np.ndarray:
    """
    Procesar imagen subida y convertirla al formato requerido con normalización de dimensiones
    
    Args:
        image_data: Datos de la imagen en bytes
        
    Returns:
        Imagen en formato numpy array (BGR) normalizada a 720x1080
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
        
        # Normalizar dimensiones a 720x1080 (vertical)
        normalized_image = normalize_image_dimensions(image_bgr, target_width=720, target_height=1080)
        
        return normalized_image
        
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

def normalize_image_dimensions(image: np.ndarray, target_width: int = 720, target_height: int = 1080) -> np.ndarray:
    """
    Normalizar imagen a dimensiones específicas (720x1080 por defecto para móvil vertical)
    
    Args:
        image: Imagen en formato numpy array (BGR)
        target_width: Ancho objetivo (720)
        target_height: Alto objetivo (1080)
        
    Returns:
        Imagen normalizada a las dimensiones especificadas
    """
    height, width = image.shape[:2]
    
    # Si la imagen es más ancha que alta, rotarla para que sea vertical
    if width > height:
        # Rotar 90 grados en sentido horario para hacer la imagen vertical
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        height, width = image.shape[:2]
    
    # Calcular la escala para ajustar manteniendo proporción
    scale_w = target_width / width
    scale_h = target_height / height
    
    # Usar la escala menor para que la imagen completa quepa
    scale = min(scale_w, scale_h)
    
    # Calcular nuevas dimensiones
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Redimensionar la imagen
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Crear imagen de fondo negro con las dimensiones objetivo
    normalized_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calcular posición para centrar la imagen redimensionada
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Colocar la imagen redimensionada en el centro
    normalized_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    
    return normalized_image
