"""
Konfigurasi untuk sistem face recognition
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATABASE_IMAGES_DIR = BASE_DIR / "database_images"
TESTING_IMAGES_DIR = BASE_DIR / "testing_images"
MODELS_DIR = BASE_DIR / "models"

# Buat folder jika belum ada
DATABASE_IMAGES_DIR.mkdir(exist_ok=True)
TESTING_IMAGES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = "face_attendance"
MONGO_COLLECTION_NAME = "face_embeddings"

# Face Detection Configuration (RetinaFace)
DETECTION_THRESHOLD = 0.5  # Confidence threshold untuk deteksi wajah
MIN_FACE_SIZE = 20  # Minimum ukuran wajah dalam pixel

# Face Recognition Configuration (ArcFace)
EMBEDDING_SIZE = 512  # Ukuran embedding vector dari ArcFace
RECOGNITION_THRESHOLD = 0.5  # Range: -1 sampai 1

# Image Configuration
TARGET_FACE_SIZE = (112, 112)  # Ukuran input untuk ArcFace

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Logging Configurations
ATTENDANCE_TIMELAPSE = 45 # absen per 45 menit
MIN_APPEARANCE_FOR_PROMOTION = 6 # minimal 6 kemunculan untuk absen (6 sks)
