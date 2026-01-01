"""
Face Detection menggunakan RetinaFace dari InsightFace library
"""
from typing import List, Tuple
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .config import DETECTION_THRESHOLD, MODELS_DIR, TARGET_FACE_SIZE


class FaceDetector:
    """
    Face Detector menggunakan RetinaFace
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Inisialisasi RetinaFace detector
        """
        print("Memuat model RetinaFace...")
        
        # Inisialisasi FaceAnalysis dengan model buffalo_l (termasuk RetinaFace + ArcFace)
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=str(MODELS_DIR),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # PENTING: Set context
        ctx_id = 0 if use_gpu else -1  # 0 = GPU, -1 = CPU
        # self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        # Prepare dengan detection size
        self.app.prepare(ctx_id=ctx_id, det_size=TARGET_FACE_SIZE, det_thresh=DETECTION_THRESHOLD)
        
        print(f"Model running on: {'GPU' if use_gpu else 'CPU'}")
        print(f"Providers: {self.app.models['recognition'].session.get_providers()}")
        
        
        print("Model RetinaFace berhasil dimuat!")
    
    def detect_faces(self, image: np.ndarray) -> list:
        """
        Deteksi wajah dalam gambar
        
        Args:
            image: Gambar dalam format numpy array (BGR)
            
        Returns:
            List of detected faces dengan informasi bbox, landmarks, embedding
        """
        if image is None:
            return []
        
        # if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Deteksi wajah
        faces = self.app.get(image)
        
        return faces
    
    def detect_single_face(self, image: np.ndarray):
        """
        Deteksi wajah tunggal (ambil yang paling besar jika ada beberapa)
        
        Args:
            image: Gambar dalam format numpy array (BGR)
            
        Returns:
            Face object atau None jika tidak ada wajah
        """
        faces = self.detect_faces(image)
        
        if not faces:
            print("[DEBUG] Tidak ada wajah terdeteksi.")
            return None
        
        if len(faces) == 1:
            return faces[0]
        
        # Jika ada beberapa wajah, ambil yang paling besar (berdasarkan area bbox)
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest_face
    
    def detect_faces_with_boxes(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Deteksi semua wajah dalam gambar dan kembalikan dengan bounding boxes
        
        Args:
            image: Gambar dalam format BGR
            
        Returns:
            Tuple (List of cropped faces, List of bounding boxes (x, y, w, h))
        """
        face_objects = []
        bboxes = []
        
        # Deteksi wajah menggunakan InsightFace
        print("Mendeteksi wajah dalam gambar...")
        detections = self.detect_faces(image)
        print(f"Ditemukan {len(detections)} wajah.")
        
        for face in detections:
            # InsightFace bbox format: [x1, y1, x2, y2]
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Convert ke format (x, y, width, height)
            w = x2 - x1
            h = y2 - y1
            
            face_objects.append(face)
            bboxes.append((x1, y1, w, h))
        
        return face_objects, bboxes