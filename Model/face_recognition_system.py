"""
Sistem Face Recognition utama yang mengintegrasikan semua komponen
"""
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from config.configrations import attendance_collection, users_collection, vector_collection
import base64
from bson import ObjectId
import time

from .config import (
    DATABASE_IMAGES_DIR,
    TESTING_IMAGES_DIR,
    RECOGNITION_THRESHOLD,
    SUPPORTED_EXTENSIONS
)
from .face_detector import FaceDetector
from .face_encoder import FaceEncoder
from .database import FaceDatabase


class FaceRecognitionSystem:
    """
    Sistem Face Recognition lengkap untuk absensi
    """
    
    def __init__(self):
        """
        Inisialisasi sistem
        """
        print("="*50)
        print("Inisialisasi Face Recognition System")
        print("="*50)
        
        self.detector = FaceDetector()
        self.encoder = FaceEncoder()
        self.database = FaceDatabase()
        
        print("="*50)
        print("Sistem siap digunakan!")
        print("="*50)
    
    def extract_name_from_filename(self, filename: str) -> str:
        """
        Ekstrak nama dari filename (format: Name_1.jpg -> Name)
        
        Args:
            filename: Nama file
            
        Returns:
            Nama orang
        """
        # Hapus ekstensi
        name_part = Path(filename).stem
        
        # Split berdasarkan underscore dan ambil semua kecuali yang terakhir (nomor)
        parts = name_part.rsplit('_', 1)
        
        if len(parts) > 1 and parts[-1].isdigit():
            return parts[0]
        
        return name_part
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load gambar dari path
        
        Args:
            image_path: Path ke gambar
            
        Returns:
            Gambar dalam format BGR atau None jika gagal
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Gagal memuat gambar {image_path}")
        return image
    
    def register_faces_from_folder(self, folder_path: str = None, clear_existing: bool = True) -> Dict:
        """
        Daftarkan semua wajah dari folder ke database
        
        Args:
            folder_path: Path ke folder berisi gambar wajah
            clear_existing: Hapus data existing sebelum registrasi
            
        Returns:
            Dictionary dengan statistik registrasi
        """
        if folder_path is None:
            folder_path = DATABASE_IMAGES_DIR
        
        folder = Path(folder_path)
        
        if not folder.exists():
            raise ValueError(f"Folder tidak ditemukan: {folder}")
        
        # Get all image files
        image_files = [f for f in folder.iterdir() 
                      if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        
        if not image_files:
            raise ValueError(f"Tidak ada gambar ditemukan di {folder}")
        
        print(f"\nMendaftarkan wajah dari: {folder}")
        print(f"Ditemukan {len(image_files)} gambar")
        
        if clear_existing:
            # deleted_vector = vector_collection.clear_database()
            vector_collection.delete_many({})
            print(f"Database dibersihkan Vektor Colllection entri dihapus)")
        
        stats = {
            "total_images": len(image_files),
            "success": 0,
            "failed": 0,
            "persons": set()
        }
        
        for image_file in tqdm(image_files, desc="Mendaftarkan wajah"):
            person_name = self.extract_name_from_filename(image_file.name)
            print(f"Memproses: {image_file.name} (Nama: {person_name})")
            # Load image
            image = self.load_image(image_file)
            if image is None:
                stats["failed"] += 1
                continue
            
            # Detect face
            face = self.detector.detect_single_face(image)
            if face is None:
                print(f"Warning: Tidak ada wajah terdeteksi di {image_file.name}")
                stats["failed"] += 1
                continue
            
            # Get embedding
            embedding = self.encoder.get_embedding(face)
            if embedding is None:
                print(f"Warning: Gagal mengekstrak embedding dari {image_file.name}")
                stats["failed"] += 1
                continue
            
            # 1. Cek apakah user dengan nama tersebut sudah ada
            existing_user = users_collection.find_one({"name": person_name})
            
            # 2. Jika belum ada, insert user baru. Jika sudah, ambil id-nya
            if existing_user is None:
                # Insert user baru
                print(f"Info: Mendaftarkan user baru '{person_name}'.")
                user_result = users_collection.insert_one({
                    "name": person_name,
                    "created_at": datetime.now().isoformat()
                })
                user_id = user_result.inserted_id
            else:
                # Ambil id user yang sudah ada
                print(f"Info: User '{person_name}' sudah terdaftar, menggunakan data existing.")
                user_id = existing_user["_id"]
            
            # 3. Insert vector dengan FK user_id
            vector_document = {
                "user_id": user_id,
                "embedding": embedding.tolist(),
                "created_at": datetime.now().isoformat()
            }
            vector_collection.insert_one(vector_document)
            
            stats["success"] += 1
            stats["persons"].add(person_name)
        
        stats["persons"] = list(stats["persons"])
        
        print(f"\n--- Hasil Registrasi ---")
        print(f"Berhasil: {stats['success']}/{stats['total_images']}")
        print(f"Gagal: {stats['failed']}/{stats['total_images']}")
        print(f"Jumlah orang terdaftar: {len(stats['persons'])}")
        print(f"Nama terdaftar: {','.join(sorted(stats['persons']))}")
        
        return stats
    
    def recognize_face(self, image: np.ndarray, threshold: float = None) -> List[Dict]:
        """
        Kenali wajah dalam gambar
        
        Args:
            image: Gambar dalam format BGR
            threshold: Threshold untuk recognition (default dari config)
            
        Returns:
            List of recognition results
        """
        if threshold is None:
            threshold = RECOGNITION_THRESHOLD
        
        results = []
        
        # Detect all faces
        face = self.detector.detect_single_face(image)
        
        embedding = self.encoder.get_embedding(face)
            
        if embedding is None:
            print("Warning: Gagal mengekstrak embedding dari gambar input")
            return results
            
        # Find closest match
        user_id, distance = self.database.find_closest_match(embedding, threshold)
        
        return user_id
    
    def recognize_from_file(self, image_path: str, threshold: float = None) -> List[Dict]:
        """
        Kenali wajah dari file gambar
        
        Args:
            image_path: Path ke gambar
            threshold: Threshold untuk recognition
            
        Returns:
            List of recognition results
        """
        image = self.load_image(image_path)
        if image is None:
            return []
        
        return self.recognize_face(image, threshold)
    
    def test_accuracy(self, test_folder: str = None, threshold: float = None) -> Dict:
        """
        Test akurasi sistem dengan folder testing
        
        Args:
            test_folder: Path ke folder testing
            threshold: Threshold untuk recognition
            
        Returns:
            Dictionary dengan metrik akurasi (accuracy, precision, recall, f1-score, FPR)
        """
        if test_folder is None:
            test_folder = TESTING_IMAGES_DIR
        
        folder = Path(test_folder)
        
        if not folder.exists():
            raise ValueError(f"Folder testing tidak ditemukan: {folder}")
        
        image_files = [f for f in folder.iterdir() 
                    if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        
        if not image_files:
            raise ValueError(f"Tidak ada gambar testing ditemukan di {folder}")
        
        print(f"\nMenguji akurasi dengan {len(image_files)} gambar testing")
        
        results = {
            "total": len(image_files),
            "correct": 0,
            "incorrect": 0,
            "no_face_detected": 0,
            "unknown_predicted": 0,
            "details": []
        }
        
        # Confusion matrix components
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        
        # Timing metrics
        prediction_times = []
        detection_times = []
        encoding_times = []
        matching_times = []
    
        # Get list of registered persons for checking if person is in database
        registered_persons = set(self.database.get_unique_persons())
        
        for image_file in tqdm(image_files, desc="Testing"):
            true_name = self.extract_name_from_filename(image_file.name)
            is_registered = true_name in registered_persons
            
            # Start total prediction timer
            total_start = time.perf_counter()
            
            # Load image
            image = self.load_image(str(image_file))
            if image is None:
                results["no_face_detected"] += 1
                results["details"].append({
                    "file": image_file.name,
                    "true_name": true_name,
                    "predicted": None,
                    "status": "no_face"
                })
                if is_registered:
                    false_negatives += 1
                continue
            
            # Step 1: Face Detection
            detection_start = time.perf_counter()
            face = self.detector.detect_single_face(image)
            detection_time = time.perf_counter() - detection_start
            detection_times.append(detection_time)
            
            if face is None:
                results["no_face_detected"] += 1
                results["details"].append({
                    "file": image_file.name,
                    "true_name": true_name,
                    "predicted": None,
                    "status": "no_face"
                })
                if is_registered:
                    false_negatives += 1
                continue
            
            # Step 2: Face Encoding
            encoding_start = time.perf_counter()
            embedding = self.encoder.get_embedding(face)
            encoding_time = time.perf_counter() - encoding_start
            encoding_times.append(encoding_time)
            
            if embedding is None:
                results["no_face_detected"] += 1
                results["details"].append({
                    "file": image_file.name,
                    "true_name": true_name,
                    "predicted": None,
                    "status": "no_face"
                })
                if is_registered:
                    false_negatives += 1
                continue
            
            # Step 3: Database Matching
            matching_start = time.perf_counter()
            user_id, distance = self.database.find_closest_match(embedding, threshold)
            matching_time = time.perf_counter() - matching_start
            matching_times.append(matching_time)
            
            # Total prediction time
            total_time = time.perf_counter() - total_start
            prediction_times.append(total_time)
            
            # Handle no match
            if user_id is None:
                results["unknown_predicted"] += 1
                results["details"].append({
                    "file": image_file.name,
                    "true_name": true_name,
                    "predicted": "Unknown",
                    "distance": distance,
                    "status": "unknown",
                    "prediction_time_ms": total_time * 1000
                })
                if is_registered:
                    false_negatives += 1
                else:
                    true_negatives += 1
                continue
            
            # Get predicted name from users_collection using user_id
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)
            
            user_doc = users_collection.find_one({"_id": user_id})
            
            if user_doc is None:
                predicted_name = "Unknown"
            else:
                predicted_name = user_doc.get("name", "Unknown")
            
            if predicted_name == "Unknown":
                results["unknown_predicted"] += 1
                status = "unknown"
                if is_registered:
                    false_negatives += 1
                else:
                    true_negatives += 1
            elif predicted_name == true_name:
                results["correct"] += 1
                status = "correct"
                true_positives += 1
            else:
                results["incorrect"] += 1
                status = "incorrect"
                false_positives += 1
            
            results["details"].append({
                "file": image_file.name,
                "true_name": true_name,
                "predicted": predicted_name,
                "distance": distance,
                "status": status,
                "prediction_time_ms": total_time * 1000
            })
        
        # Calculate metrics
        recognized = results["correct"] + results["incorrect"]
        if recognized > 0:
            results["accuracy"] = results["correct"] / recognized * 100
        else:
            results["accuracy"] = 0
        
        # Precision = TP / (TP + FP)
        if (true_positives + false_positives) > 0:
            results["precision"] = true_positives / (true_positives + false_positives) * 100
        else:
            results["precision"] = 0
        
        # Recall = TP / (TP + FN)
        if (true_positives + false_negatives) > 0:
            results["recall"] = true_positives / (true_positives + false_negatives) * 100
        else:
            results["recall"] = 0
        
        # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        if (results["precision"] + results["recall"]) > 0:
            results["f1_score"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"])
        else:
            results["f1_score"] = 0
        
        # False Positive Rate (FPR) = FP / (FP + TN)
        if (false_positives + true_negatives) > 0:
            results["false_positive_rate"] = false_positives / (false_positives + true_negatives) * 100
        else:
            total_predictions = true_positives + false_positives + false_negatives + true_negatives
            if total_predictions > 0:
                results["false_positive_rate"] = false_positives / total_predictions * 100
            else:
                results["false_positive_rate"] = 0
        
        # Store confusion matrix values
        results["confusion_matrix"] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives
        }
        
        # Calculate timing statistics
        if prediction_times:
            results["timing"] = {
                "total_prediction": {
                    "mean_ms": np.mean(prediction_times) * 1000,
                    "min_ms": np.min(prediction_times) * 1000,
                    "max_ms": np.max(prediction_times) * 1000,
                    "std_ms": np.std(prediction_times) * 1000,
                    "fps": 1.0 / np.mean(prediction_times) if np.mean(prediction_times) > 0 else 0
                },
                "detection": {
                    "mean_ms": np.mean(detection_times) * 1000 if detection_times else 0,
                    "percentage": (np.mean(detection_times) / np.mean(prediction_times) * 100) if prediction_times else 0
                },
                "encoding": {
                    "mean_ms": np.mean(encoding_times) * 1000 if encoding_times else 0,
                    "percentage": (np.mean(encoding_times) / np.mean(prediction_times) * 100) if prediction_times else 0
                },
                "matching": {
                    "mean_ms": np.mean(matching_times) * 1000 if matching_times else 0,
                    "percentage": (np.mean(matching_times) / np.mean(prediction_times) * 100) if prediction_times else 0
                }
            }
        else:
            results["timing"] = None
        
        # Print results
        print(f"\n--- Hasil Testing ---")
        print(f"Total gambar: {results['total']}")
        print(f"Benar: {results['correct']}")
        print(f"Salah: {results['incorrect']}")
        print(f"Unknown: {results['unknown_predicted']}")
        print(f"Tidak ada wajah: {results['no_face_detected']}")
        print(f"\n--- Metrik Performa ---")
        print(f"Akurasi: {results['accuracy']:.2f}%")
        print(f"Precision: {results['precision']:.2f}%")
        print(f"Recall: {results['recall']:.2f}%")
        print(f"F1-Score: {results['f1_score']:.2f}%")
        print(f"False Positive Rate: {results['false_positive_rate']:.2f}%")
        
        # Print timing results
        if results["timing"]:
            print(f"\n--- Kecepatan Prediksi ---")
            timing = results["timing"]
            print(f"Rata-rata waktu per wajah: {timing['total_prediction']['mean_ms']:.2f} ms")
            print(f"Waktu tercepat: {timing['total_prediction']['min_ms']:.2f} ms")
            print(f"Waktu terlambat: {timing['total_prediction']['max_ms']:.2f} ms")
            print(f"Standar deviasi: {timing['total_prediction']['std_ms']:.2f} ms")
            print(f"FPS (Frames Per Second): {timing['total_prediction']['fps']:.2f}")
            print(f"\n--- Breakdown Waktu ---")
            print(f"Detection: {timing['detection']['mean_ms']:.2f} ms ({timing['detection']['percentage']:.1f}%)")
            print(f"Encoding: {timing['encoding']['mean_ms']:.2f} ms ({timing['encoding']['percentage']:.1f}%)")
            print(f"Matching: {timing['matching']['mean_ms']:.2f} ms ({timing['matching']['percentage']:.1f}%)")
        
        # Check against requirements
        print(f"\n--- Evaluasi Target ---")
        accuracy_pass = results['accuracy'] >= 98
        fpr_pass = results['false_positive_rate'] < 1
        speed_pass = results["timing"] and results["timing"]["total_prediction"]["mean_ms"] < 1000  # < 1 detik
        print(f"Akurasi >= 98%: {'✓ PASS' if accuracy_pass else '✗ FAIL'} ({results['accuracy']:.2f}%)")
        print(f"FPR < 1%: {'✓ PASS' if fpr_pass else '✗ FAIL'} ({results['false_positive_rate']:.2f}%)")
        if results["timing"]:
            print(f"Kecepatan < 1 detik: {'✓ PASS' if speed_pass else '✗ FAIL'} ({results['timing']['total_prediction']['mean_ms']:.2f} ms)")
        
        print(f"\n--- Confusion Matrix ---")
        print(f"True Positives (TP): {true_positives}")
        print(f"False Positives (FP): {false_positives}")
        print(f"False Negatives (FN): {false_negatives}")
        print(f"True Negatives (TN): {true_negatives}")
        
        # Print details
        print(f"\n--- Detail ---")
        for detail in results["details"]:
            status_icon = "✓" if detail["status"] == "correct" else "✗" if detail["status"] == "incorrect" else "?"
            distance_str = f"(dist: {detail.get('distance', 'N/A'):.3f})" if detail.get('distance') else ""
            time_str = f"[{detail.get('prediction_time_ms', 0):.1f}ms]" if detail.get('prediction_time_ms') else ""
            print(f"{status_icon} {detail['file']}: {detail['true_name']} -> {detail['predicted']} {distance_str} {time_str}")
    
        return results
    
    def get_database_stats(self) -> Dict:
        """
        Ambil statistik database
        
        Returns:
            Dictionary dengan statistik
        """
        persons = self.database.get_unique_persons()
        
        stats = {
            "total_embeddings": self.database.get_embedding_count(),
            "total_persons": len(persons),
            "persons": {}
        }
        
        for person in persons:
            embeddings = self.database.get_embeddings_by_person(person)
            stats["persons"][person] = len(embeddings)
        
        return stats
    
    def load_image_from_base64(self, image_base64: str) -> Optional[np.ndarray]:
        """
        Load gambar dari Base64 string
        
        Args:
            image_base64: String Base64 dari gambar
            
        Returns:
            Gambar dalam format BGR (sama seperti cv2.imread) atau None jika gagal
        """
        try:
            # Decode base64 ke bytes
            image_bytes = base64.b64decode(image_base64)
            
            # Convert bytes ke numpy array
            np_array = np.frombuffer(image_bytes, dtype=np.uint8)
            
            # Decode numpy array ke image BGR (sama seperti cv2.imread)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print("Warning: Gagal decode gambar dari base64")
                return None
            
            print(f"[DEBUG] Loaded image from base64, shape: {image.shape}")
            return image
            
        except Exception as e:
            print(f"Error: Gagal memproses base64 image: {e}")
            return None
    
    def recognize_from_base64(self, image_base64: str, threshold: float = None) -> List[Dict]:
        """
        Kenali wajah dari Base64 string
        
        Args:
            image_base64: String Base64 dari gambar
            threshold: Threshold untuk recognition
            
        Returns:
            List of recognition results
        """
        image = self.load_image_from_base64(image_base64)
        if image is None:
            return []
        return self.recognize_face(image, threshold)
    
    
    def close(self):
        """
        Tutup sistem
        """
        self.database.close()
        