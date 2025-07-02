import cv2
import numpy as np
import mediapipe as mp
from typing import Dict
import logging
import time

logger = logging.getLogger(__name__)

class FaceDistanceDetector:
    """Detector independiente para medir distancia facial usando MediaPipe"""
    
    def __init__(self):
        """Inicializar MediaPipe FaceMesh para detección de distancia"""
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Configurar MediaPipe FaceMesh específicamente para distancia
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        logger.info("FaceDistanceDetector inicializado correctamente")
    
    def detect_distance(self, image: np.ndarray) -> Dict:
        """
        Detectar la distancia de la cara basada en el valor Z de la nariz
        
        Args:
            image: Imagen en formato BGR
            
        Returns:
            Diccionario con información completa de distancia
        """
        start_time = time.time()
        
        try:
            # Convertir BGR a RGB para MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Procesar imagen con FaceMesh
            results = self.face_mesh.process(image_rgb)
            
            processing_time = time.time() - start_time
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Obtener el valor Z de la nariz (landmark 1)
                    nose_z = face_landmarks.landmark[1].z
                    
                    # Determinar categoría y obtener información completa
                    distance_info = self._analyze_distance(nose_z)
                    
                    return {
                        "success": True,
                        "face_detected": True,
                        "z_value": float(nose_z),
                        "distance_category": distance_info["category"],
                        "distance_description": distance_info["description"],
                        "distance_status": distance_info["status"],
                        "recommendation": distance_info["recommendation"],
                        "is_optimal": distance_info["is_optimal"],
                        "processing_time": processing_time
                    }
            
            # No se detectó cara
            return {
                "success": True,
                "face_detected": False,
                "z_value": None,
                "distance_category": "no_face",
                "distance_description": "No se detectó rostro en la imagen",
                "distance_status": "unknown",
                "recommendation": "Asegúrate de que tu rostro esté visible en la cámara",
                "is_optimal": False,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error en detección de distancia facial: {e}")
            
            return {
                "success": False,
                "face_detected": False,
                "z_value": None,
                "distance_category": "error",
                "distance_description": f"Error en la detección: {str(e)}",
                "distance_status": "error",
                "recommendation": "Intenta nuevamente con una imagen clara",
                "is_optimal": False,
                "processing_time": processing_time,
                "error": str(e)
            }
    
    def _analyze_distance(self, z_value: float) -> Dict:
        """
        Analizar el valor Z y proporcionar información detallada
        
        Args:
            z_value: Valor Z de la nariz
            
        Returns:
            Diccionario con análisis completo de la distancia
        """
        if z_value < -0.08:
            return {
                "category": "muy_cerca",
                "description": "Muy cerca - Aléjate un poco",
                "status": "too_close",
                "recommendation": "Muévete hacia atrás para obtener mejor detección",
                "is_optimal": False
            }
        elif -0.08 <= z_value <= -0.06:
            return {
                "category": "normal",
                "description": "Distancia normal - Bien",
                "status": "acceptable",
                "recommendation": "Distancia aceptable para la detección",
                "is_optimal": False
            }
        elif -0.05 <= z_value <= -0.03:
            return {
                "category": "perfecta",
                "description": "Distancia perfecta - Excelente",
                "status": "optimal",
                "recommendation": "¡Perfecto! Mantén esta distancia",
                "is_optimal": True
            }
        else:  # z_value > -0.02
            return {
                "category": "muy_lejos",
                "description": "Muy lejos - Acércate más",
                "status": "too_far",
                "recommendation": "Acércate más a la cámara para mejor detección",
                "is_optimal": False
            }
    
    def get_distance_thresholds(self) -> Dict:
        """
        Obtener los umbrales de distancia configurados
        
        Returns:
            Diccionario con los umbrales y sus descripciones
        """
        return {
            "thresholds": {
                "muy_cerca": {"min": float('-inf'), "max": -0.08},
                "normal": {"min": -0.08, "max": -0.06},
                "perfecta": {"min": -0.05, "max": -0.03},
                "muy_lejos": {"min": -0.02, "max": float('inf')}
            },
            "optimal_range": {
                "min": -0.05,
                "max": -0.03,
                "description": "Rango óptimo para mejor detección"
            },
            "recommendations": {
                "muy_cerca": "Aléjate de la cámara",
                "normal": "Distancia aceptable",
                "perfecta": "Distancia ideal - mantén la posición",
                "muy_lejos": "Acércate a la cámara"
            }
        }
    
    def is_distance_optimal(self, z_value: float) -> bool:
        """
        Verificar si la distancia es óptima
        
        Args:
            z_value: Valor Z de la nariz
            
        Returns:
            True si la distancia es óptima
        """
        return -0.05 <= z_value <= -0.03
    
    def __del__(self):
        """Limpiar recursos"""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
        except:
            pass
