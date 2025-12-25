"""
Face Detection menggunakan RetinaFace dari InsightFace library
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .config import DETECTION_THRESHOLD, MODELS_DIR


class FaceDetector:
    """
    Face Detector menggunakan RetinaFace
    """
    
    def __init__(self):
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
        
        # Prepare dengan detection size
        self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=DETECTION_THRESHOLD)
        
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
    
    def get_face_image(self, image: np.ndarray, face, margin: float = 0.2) -> np.ndarray:
        """
        Crop gambar wajah dari gambar asli
        
        Args:
            image: Gambar asli
            face: Face object dari detector
            margin: Margin tambahan di sekitar wajah
            
        Returns:
            Cropped face image
        """
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Tambahkan margin
        w, h = x2 - x1, y2 - y1
        margin_x, margin_y = int(w * margin), int(h * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)
        
        return image[y1:y2, x1:x2]
    
    def draw_detection(self, image: np.ndarray, face, name: str = None, distance: float = None) -> np.ndarray:
        """
        Gambar bounding box dan informasi pada gambar
        
        Args:
            image: Gambar asli
            face: Face object
            name: Nama orang (jika sudah dikenali)
            distance: Euclidean distance
            
        Returns:
            Gambar dengan anotasi
        """
        img_copy = image.copy()
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Warna berdasarkan apakah dikenali atau tidak
        color = (0, 255, 0) if name and name != "Unknown" else (0, 0, 255)
        
        # Gambar bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Gambar label
        if name:
            label = f"{name}"
            if distance is not None:
                label += f" ({distance:.2f})"
            
            # Background untuk text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img_copy