import numpy as np
import tensorflow as tf
import cv2
import time
import os
from typing import Dict, List, Optional, Tuple
import logging

from utils.mediapipe_processor import MediaPipeProcessor

logger = logging.getLogger(__name__)

class GesturePredictor:
    """Clase principal para predicción de gestos estáticos"""
    
    def __init__(self, models_path: str = "models/saved_models"):
        """
        Inicializar el predictor de gestos
        
        Args:
            models_path: Ruta donde están los modelos guardados
        """
        self.models_path = models_path
        self.mediapipe_processor = MediaPipeProcessor()
        
        # Modelos y clases
        self.modelo_una_mano = None
        self.modelo_dos_manos = None
        self.clases_una_mano = None
        self.clases_dos_manos = None
        
        # Configuración
        self.umbral_confianza = 0.5
        
        # Cargar modelos
        self._load_models()
    
    def _load_models(self):
        """Cargar los modelos entrenados y las clases"""
        try:
            # Cargar modelo de una mano
            model_path_one = os.path.join(self.models_path, "modelo_estatico_una_mano.h5")
            classes_path_one = os.path.join(self.models_path, "clases_estatico_una_mano.npy")
            
            if os.path.exists(model_path_one) and os.path.exists(classes_path_one):
                self.modelo_una_mano = tf.keras.models.load_model(model_path_one)
                self.clases_una_mano = np.load(classes_path_one, allow_pickle=True)
                # Quitar acentos de las clases
                self.clases_una_mano = np.array([self._quitar_acentos(clase) for clase in self.clases_una_mano])
                logger.info(f"Modelo de una mano cargado: {len(self.clases_una_mano)} clases")
            else:
                logger.warning("No se encontró el modelo de una mano")
            
            # Cargar modelo de dos manos
            model_path_two = os.path.join(self.models_path, "modelo_estatico_dos_manos.h5")
            classes_path_two = os.path.join(self.models_path, "clases_estatico_dos_manos.npy")
            
            if os.path.exists(model_path_two) and os.path.exists(classes_path_two):
                self.modelo_dos_manos = tf.keras.models.load_model(model_path_two)
                self.clases_dos_manos = np.load(classes_path_two, allow_pickle=True)
                # Quitar acentos de las clases
                self.clases_dos_manos = np.array([self._quitar_acentos(clase) for clase in self.clases_dos_manos])
                logger.info(f"Modelo de dos manos cargado: {len(self.clases_dos_manos)} clases")
            else:
                logger.warning("No se encontró el modelo de dos manos")
                
        except Exception as e:
            logger.error(f"Error al cargar modelos: {e}")
    
    def _quitar_acentos(self, texto: str) -> str:
        """Quitar acentos del texto"""
        mapeo = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ü': 'u', 'Ü': 'U', 'ñ': 'n', 'Ñ': 'N'
        }
        
        for acentuado, no_acentuado in mapeo.items():
            texto = texto.replace(acentuado, no_acentuado)
        
        return texto
    
    def _normalizar_caracteristicas(self, caracteristicas: np.ndarray, tamano_esperado: int) -> np.ndarray:
        """
        Normalizar características para que coincidan con el tamaño esperado
        
        Args:
            caracteristicas: Vector de características
            tamano_esperado: Tamaño esperado por el modelo
            
        Returns:
            Vector normalizado
        """
        if caracteristicas.shape[0] == tamano_esperado:
            return caracteristicas
        
        if caracteristicas.shape[0] < tamano_esperado:
            # Rellenar con ceros
            return np.pad(caracteristicas, (0, tamano_esperado - caracteristicas.shape[0]), 'constant')
        else:
            # Truncar
            return caracteristicas[:tamano_esperado]
    
    def predict_from_image(self, image: np.ndarray) -> Dict:
        """
        Realizar predicción a partir de una imagen
        
        Args:
            image: Imagen en formato numpy array (BGR)
            
        Returns:
            Diccionario con la predicción y metadatos
        """
        start_time = time.time()
        
        try:
            # Procesar imagen con MediaPipe
            landmarks_data = self.mediapipe_processor.process_image(image)
            
            puntos_manos = landmarks_data["hands"]
            puntos_cara = landmarks_data["face"]
            puntos_hombros = landmarks_data["shoulders"]
            
            num_hands = len(puntos_manos)
            
            # Determinar qué modelo usar basado en el número de manos
            if num_hands == 0:
                return {
                    "prediction": "?",
                    "confidence": 0.0,
                    "num_hands": 0,
                    "processing_time": time.time() - start_time,
                    "error": "No se detectaron manos"
                }
            
            elif num_hands == 1 and self.modelo_una_mano is not None:
                # Usar modelo de una mano
                prediction, confidence = self._predict_one_hand(puntos_manos, puntos_cara, puntos_hombros)
                
            elif num_hands >= 2 and self.modelo_dos_manos is not None:
                # Usar modelo de dos manos
                prediction, confidence = self._predict_two_hands(puntos_manos, puntos_cara, puntos_hombros)
                
            else:
                # No hay modelo disponible para este caso
                return {
                    "prediction": "?",
                    "confidence": 0.0,
                    "num_hands": num_hands,
                    "processing_time": time.time() - start_time,
                    "error": f"No hay modelo disponible para {num_hands} mano(s)"
                }
            
            processing_time = time.time() - start_time
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "num_hands": num_hands,
                "processing_time": processing_time,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return {
                "prediction": "?",
                "confidence": 0.0,
                "num_hands": 0,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _predict_one_hand(self, puntos_manos: List, puntos_cara: List, puntos_hombros: List) -> Tuple[str, float]:
        """Predicción para una mano"""
        try:
            # Preparar características
            caracteristicas = np.array(puntos_manos[0]).flatten()
            
            # Añadir puntos faciales y de hombros si existen
            if puntos_cara:
                puntos_cara_planos = np.array(puntos_cara).flatten()
                caracteristicas = np.concatenate([caracteristicas, puntos_cara_planos])
            
            if puntos_hombros:
                puntos_hombros_planos = np.array(puntos_hombros).flatten()
                caracteristicas = np.concatenate([caracteristicas, puntos_hombros_planos])
            
            # Normalizar al tamaño esperado
            input_shape = self.modelo_una_mano.input_shape[1]
            caracteristicas = self._normalizar_caracteristicas(caracteristicas, input_shape)
            
            # Realizar predicción
            prediccion = self.modelo_una_mano.predict(np.array([caracteristicas]), verbose=0)
            clase_idx = np.argmax(prediccion[0])
            confianza = prediccion[0][clase_idx]
            
            if confianza >= self.umbral_confianza:
                return self.clases_una_mano[clase_idx], confianza
            else:
                return "?", confianza
                
        except Exception as e:
            logger.error(f"Error en predicción de una mano: {e}")
            return "?", 0.0
    
    def _predict_two_hands(self, puntos_manos: List, puntos_cara: List, puntos_hombros: List) -> Tuple[str, float]:
        """Predicción para dos manos"""
        try:
            # Preparar características
            puntos_mano1 = np.array(puntos_manos[0]).flatten()
            puntos_mano2 = np.array(puntos_manos[1]).flatten()
            caracteristicas = np.concatenate([puntos_mano1, puntos_mano2])
            
            # Añadir puntos faciales y de hombros si existen
            if puntos_cara:
                puntos_cara_planos = np.array(puntos_cara).flatten()
                caracteristicas = np.concatenate([caracteristicas, puntos_cara_planos])
            
            if puntos_hombros:
                puntos_hombros_planos = np.array(puntos_hombros).flatten()
                caracteristicas = np.concatenate([caracteristicas, puntos_hombros_planos])
            
            # Normalizar al tamaño esperado
            input_shape = self.modelo_dos_manos.input_shape[1]
            caracteristicas = self._normalizar_caracteristicas(caracteristicas, input_shape)
            
            # Realizar predicción
            prediccion = self.modelo_dos_manos.predict(np.array([caracteristicas]), verbose=0)
            clase_idx = np.argmax(prediccion[0])
            confianza = prediccion[0][clase_idx]
            
            if confianza >= self.umbral_confianza:
                return self.clases_dos_manos[clase_idx], confianza
            else:
                return "?", confianza
                
        except Exception as e:
            logger.error(f"Error en predicción de dos manos: {e}")
            return "?", 0.0
    
    def check_models_status(self) -> Dict:
        """Verificar el estado de los modelos"""
        return {
            "one_hand_model": self.modelo_una_mano is not None,
            "two_hands_model": self.modelo_dos_manos is not None,
            "one_hand_classes": len(self.clases_una_mano) if self.clases_una_mano is not None else 0,
            "two_hands_classes": len(self.clases_dos_manos) if self.clases_dos_manos is not None else 0,
            "all_loaded": self.modelo_una_mano is not None and self.modelo_dos_manos is not None
        }
    
    def get_available_gestures(self) -> Dict:
        """Obtener gestos disponibles"""
        return {
            "one_hand": self.clases_una_mano.tolist() if self.clases_una_mano is not None else [],
            "two_hands": self.clases_dos_manos.tolist() if self.clases_dos_manos is not None else []
        }
    
    def get_model_info(self) -> Dict:
        """Obtener información detallada de los modelos"""
        info = {
            "models_loaded": self.check_models_status(),
            "confidence_threshold": self.umbral_confianza,
            "mediapipe_version": "0.10.0"  # Ajustar según tu versión
        }
        
        if self.modelo_una_mano is not None:
            info["one_hand_model_info"] = {
                "input_shape": self.modelo_una_mano.input_shape,
                "output_shape": self.modelo_una_mano.output_shape,
                "total_params": self.modelo_una_mano.count_params()
            }
        
        if self.modelo_dos_manos is not None:
            info["two_hands_model_info"] = {
                "input_shape": self.modelo_dos_manos.input_shape,
                "output_shape": self.modelo_dos_manos.output_shape,
                "total_params": self.modelo_dos_manos.count_params()
            }
        
        return info
    
    def set_confidence_threshold(self, threshold: float):
        """Cambiar el umbral de confianza"""
        self.umbral_confianza = max(0.1, min(0.9, threshold))
