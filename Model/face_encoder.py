"""
Face Encoding menggunakan ArcFace dari InsightFace library
"""
import numpy as np
from typing import Optional


class FaceEncoder:
    """
    Face Encoder menggunakan ArcFace
    Embedding sudah terintegrasi dalam FaceAnalysis dari InsightFace
    """
    
    def __init__(self):
        """
        Inisialisasi - embedding sudah dilakukan oleh FaceDetector
        """
        print("FaceEncoder siap!  (Menggunakan embedding dari InsightFace)")
    
    def get_embedding(self, face) -> Optional[np.ndarray]:
        """
        Ambil embedding dari face object
        
        Args:
            face: Face object dari InsightFace detector
            
        Returns:
            Embedding vector (512-dimensional) atau None
        """
        if face is None:
            return None
        
        # InsightFace sudah menyediakan embedding dalam face object
        embedding = face.embedding
        
        if embedding is None:
            return None
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    @staticmethod
    def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Hitung Cosine similarity antara dua embedding
        
        Args:
            embedding1: Embedding pertama
            embedding2: Embedding kedua
            
        Returns:
            Cosine similarity (semakin besar = semakin mirip, range -1 to 1)
        """
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))