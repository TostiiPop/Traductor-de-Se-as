import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MediaPipeProcessor:
    """Procesador de MediaPipe para extracción de landmarks de gestos de manos"""
    
    def __init__(self):
        """Inicializar MediaPipe para detección de gestos"""
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        
        # Configurar MediaPipe con mayor sensibilidad
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            model_complexity=1
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=True
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_complexity=1
        )
    
    def process_image(self, image: np.ndarray) -> Dict:
        """
        Procesar imagen y extraer landmarks para gestos
        
        Args:
            image: Imagen en formato BGR
            
        Returns:
            Diccionario con landmarks extraídos
        """
        try:
            # Convertir BGR a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            resultados_manos = self.hands.process(image_rgb)
            resultados_cara = self.face_mesh.process(image_rgb)
            resultados_pose = self.pose.process(image_rgb)
            
            # Extraer puntos de manos
            puntos_manos = self._extract_hand_landmarks(resultados_manos)
            
            # Extraer puntos faciales
            puntos_cara = self._extract_face_landmarks(resultados_cara)
            
            # Extraer puntos de hombros
            puntos_hombros = self._extract_shoulder_landmarks(resultados_pose)
            
            return {
                "hands": puntos_manos,
                "face": puntos_cara,
                "shoulders": puntos_hombros,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error al procesar imagen con MediaPipe: {e}")
            return {
                "hands": [],
                "face": [],
                "shoulders": [],
                "success": False,
                "error": str(e)
            }
    
    def _extract_hand_landmarks(self, resultados_manos) -> List:
        """Extraer landmarks de las manos"""
        puntos_manos = []
        
        if resultados_manos.multi_hand_landmarks:
            for hand_landmarks in resultados_manos.multi_hand_landmarks:
                # Extraer coordenadas normalizadas
                puntos = []
                for punto in hand_landmarks.landmark:
                    x, y, z = punto.x, punto.y, punto.z
                    puntos.append([x, y, z])
                
                # Convertir a numpy array
                puntos = np.array(puntos)
                
                # Normalizar orientación de la mano
                es_mano_izquierda = puntos[4][0] < puntos[17][0]
                
                if es_mano_izquierda:
                    for i in range(len(puntos)):
                        puntos[i][0] = -puntos[i][0]
                
                # Normalizar posición (centrar en la muñeca)
                puntos = puntos - puntos[0]
                
                # Normalizar escala
                distancia_max = np.max(np.linalg.norm(puntos, axis=1))
                if distancia_max > 0:
                    puntos = puntos / distancia_max
                
                puntos_manos.append(puntos.tolist())
        
        return puntos_manos
    
    def _extract_face_landmarks(self, resultados_cara) -> List:
        """Extraer landmarks faciales de interés para gestos"""
        puntos_cara = []
        
        if resultados_cara.multi_face_landmarks:
            for face_landmarks in resultados_cara.multi_face_landmarks:
                # Índices de puntos de interés: nariz, ojos, orejas
                indices_interes = [1, 33, 263, 61, 291]
                
                for idx in indices_interes:
                    punto = face_landmarks.landmark[idx]
                    x, y, z = punto.x, punto.y, punto.z
                    puntos_cara.append([x, y, z])
                
                break  # Solo procesar la primera cara
        
        return puntos_cara
    
    def _extract_shoulder_landmarks(self, resultados_pose) -> List:
        """Extraer landmarks de los hombros"""
        puntos_hombros = []
        
        if resultados_pose.pose_landmarks:
            # Índices de los hombros en MediaPipe Pose
            indices_hombros = [11, 12]  # Hombro izquierdo, hombro derecho
            
            for idx in indices_hombros:
                punto = resultados_pose.pose_landmarks.landmark[idx]
                if punto.visibility > 0.5:  # Solo si es visible
                    x, y, z = punto.x, punto.y, punto.z
                    puntos_hombros.append([x, y, z])
        
        return puntos_hombros
    
    def __del__(self):
        """Limpiar recursos"""
        try:
            if hasattr(self, 'hands'):
                self.hands.close()
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            if hasattr(self, 'pose'):
                self.pose.close()
        except:
            pass
