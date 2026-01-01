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

CHOCK_POINT_IMAGES = BASE_DIR / "ChokePoint"

# Open-Set Evaluation paths
REGISTERED_USERS_DIR = BASE_DIR / "registered_users"
UNKNOWN_PEOPLE_DIR = BASE_DIR / "unknown_people"

# Buat folder jika belum ada
DATABASE_IMAGES_DIR.mkdir(exist_ok=True)
TESTING_IMAGES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Mongodb configuration is in config/configrations.py

# Face Detection Configuration (RetinaFace)
DETECTION_THRESHOLD = 0.5  # Confidence threshold untuk deteksi wajah
MIN_FACE_SIZE = 20  # Minimum ukuran wajah dalam pixel

# Face Recognition Configuration (ArcFace)
EMBEDDING_SIZE = 512  # Ukuran embedding vector dari ArcFace
RECOGNITION_THRESHOLD = 0.5  # Range: -1 sampai 1

# Open-Set Evaluation Configuration
DEFAULT_THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]  # Thresholds untuk ROC curve

# Image Configuration
TARGET_FACE_SIZE = (224, 224)  # Ukuran input untuk ArcFace

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Logging Configurations
ATTENDANCE_TIMELAPSE = 45 # absen per 45 menit
MIN_APPEARANCE_FOR_PROMOTION = 6 # minimal 6 kemunculan untuk absen (6 sks)
