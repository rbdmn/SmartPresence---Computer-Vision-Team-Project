"""
Sistem Face Recognition utama yang mengintegrasikan semua komponen
"""
from datetime import datetime, timedelta
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
import json
from tqdm import tqdm
from config.configrations import attendance_collection, users_collection, vector_collection, visitor_vector_collection, visitor_collection
import base64
from bson import ObjectId
import time
from .timer import timerawal, timerstoptampilkantimer

from .config import (
    DATABASE_IMAGES_DIR,
    TESTING_IMAGES_DIR,
    RECOGNITION_THRESHOLD,
    SUPPORTED_EXTENSIONS,
    MIN_APPEARANCE_FOR_PROMOTION,
    CHOCK_POINT_IMAGES,
    REGISTERED_USERS_DIR,
    UNKNOWN_PEOPLE_DIR,
    DEFAULT_THRESHOLDS
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
        
        # Cache embeddings saat startup (bukan per-request)
        print("Loading embeddings ke cache...")
        self._cached_embeddings = []
        self._cached_visitor_embeddings = []
        self.refresh_embeddings_cache()
                
        print("="*50)
        print("Sistem siap digunakan!") 
        print("="*50)
    
    def refresh_embeddings_cache(self):
        """
        Refresh cache embeddings dari database.
        Panggil method ini setelah ada perubahan data embedding.
        """
        print("Refreshing embeddings cache...")
        self._cached_embeddings = self.database.get_all_embeddings()
        self._cached_visitor_embeddings = self.database.get_all_visitor_embeddings()
        print(f"Cache updated: {len(self._cached_embeddings)} user embeddings, "
              f"{len(self._cached_visitor_embeddings)} visitor embeddings")
    
    def extract_name_from_filename(self, filename: str) -> str:
        """
        Ekstrak nama dari filename.
        - Format lama: Name_1.jpg -> Name
        - Dukungan tambahan: 10_far, 10_mid, 10_near, 1_near, 2_far, dll -> kembalikan angka depan (kelas)
        """
        # Hapus ekstensi
        name_part = Path(filename).stem

        # Hapus suffix posisi jika ada (_far, _mid, _near) - case insensitive
        lowered = name_part.lower()
        for suffix in ("_far", "_mid", "_near"):
            if lowered.endswith(suffix):
                name_part = name_part[: len(name_part) - len(suffix)]
                break

        # Jika format Name_1 atau Name_1_anything, ambil bagian sebelum angka terakhir
        parts = name_part.rsplit("_", 1)
        if len(parts) > 1 and parts[-1].isdigit():
            return parts[0]

        # Jika seluruh stem adalah angka (contoh: "10" atau "2"), kembalikan angka itu
        if name_part.isdigit():
            return name_part

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
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            folder_path = CHOCK_POINT_IMAGES
        
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
        
        # Refresh cache setelah registrasi selesai
        self.refresh_embeddings_cache()
        
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
            
        # Find closest match (gunakan cached embeddings)
        user_id, distance = self.database.find_closest_match(embedding, threshold, self._cached_embeddings)
        
        return user_id
    
    def recognize_faces(self, image: np.ndarray, class_id: str, threshold: float = None) -> List[Dict]:
        """
        Kenali banyak wajah dalam gambar
        
        Args:
            image: Gambar dalam format BGR
            threshold: Threshold untuk recognition (default dari config)
            
        Returns:
            List of recognition results dengan user_id dan bounding box
        """
        timer_awal = time.perf_counter()
        if threshold is None:
            threshold = RECOGNITION_THRESHOLD
        
        results = []
        
        # Detect all faces with bounding boxes
        timer_deteksi_wajah = time.perf_counter()
        faces, bboxes = self.detector.detect_faces_with_boxes(image)
        print("WAKTU: Waktu deteksi wajah:", time.perf_counter() - timer_deteksi_wajah)
        
        if not faces:
            print("Warning: Tidak ada wajah terdeteksi")
            return results
        
        # Gunakan cached embeddings (sudah di-load saat startup)
        all_embeddings = self._cached_embeddings
        all_visitor_embeddings = self._cached_visitor_embeddings
        print(f"Debug: Total embeddings di cache: {len(all_embeddings)}")
        
        timer_proses_wajah = time.perf_counter()
        for i, face in enumerate(faces):
            embedding = self.encoder.get_embedding(face)
            
            if embedding is None:
                print(f"Warning: Gagal mengekstrak embedding dari wajah ke-{i+1}")
                continue
            
            # Find closest match
            print("Info: Mencari kecocokan untuk wajah ke-", i+1)
            timer_cari_cocok = time.perf_counter()
            user_id, distance = self.database.find_closest_match(embedding, all_embeddings, threshold)
            print("WAKTU: Waktu cari kecocokan untuk wajah ke-", i+1, ":", time.perf_counter() - timer_cari_cocok)
            
            timer_handle_match = time.perf_counter()
            if user_id: # jika ditemukan di database users
                added = self.database.maybe_add_embedding(user_id, embedding, all_embeddings)
                if added:
                    # Refresh cache karena ada embedding baru
                    self.refresh_embeddings_cache()
                    all_embeddings = self._cached_embeddings
                # Get bounding box
                bbox = bboxes[i] if i < len(bboxes) else None
                results.append({
                    "user_id": user_id,
                    "distance": distance,
                    "bounding_box": {
                        "x": int(bbox[0]) if bbox is not None else None,
                        "y": int(bbox[1]) if bbox is not None else None,
                        "width": int(bbox[2]) if bbox is not None else None,
                        "height": int(bbox[3]) if bbox is not None else None,
                        "x2": int(bbox[0] + bbox[2]) if bbox is not None else None,
                        "y2": int(bbox[1] + bbox[3]) if bbox is not None else None
                    } if bbox is not None else None
                })
                self.database.add_user_attendance(user_id, class_id)
            else: # jika tidak ditemukan di database users, cek di visitor
                print("Info: Wajah tidak dikenali, memeriksa di database visitor...")
                visitor_id, distance = self.database.find_visitor_closest_match(embedding, all_visitor_embeddings)
                if visitor_id:
                    self.database.add_visitor_attendance(visitor_id, class_id)
                    print("Info: Wajah dikenali sebagai visitor dengan ID:", visitor_id)
                    bbox = bboxes[i] if i < len(bboxes) else None
                    print("Info: Memperbarui data visitor...")
                    self.handle_known_visitor(visitor_id, embedding, all_visitor_embeddings)
                    print("Info: Data visitor diperbarui.")
                    results.append({
                        "visitor_id": visitor_id,
                        "distance": distance,
                        "bounding_box": {
                            "x": int(bbox[0]) if bbox is not None else None,
                            "y": int(bbox[1]) if bbox is not None else None,
                            "width": int(bbox[2]) if bbox is not None else None,
                            "height": int(bbox[3]) if bbox is not None else None,
                            "x2": int(bbox[0] + bbox[2]) if bbox is not None else None,
                            "y2": int(bbox[1] + bbox[3]) if bbox is not None else None
                        } if bbox is not None else None
                    })
                    
                else: # jika di visitor juga tidak ditemukan, tambahkan sebagai visitor baru
                    visitor_id = self.database.add_new_visitor(embedding)
                    # Refresh cache karena visitor baru ditambahkan
                    self.refresh_embeddings_cache()
                    all_visitor_embeddings = self._cached_visitor_embeddings
                    self.database.add_visitor_attendance(visitor_id, class_id)
                    bbox = bboxes[i] if i < len(bboxes) else None
                    results.append({
                        "visitor_id": visitor_id,
                        "distance": None,
                        "bounding_box": {
                            "x": int(bbox[0]) if bbox is not None else None,
                            "y": int(bbox[1]) if bbox is not None else None,
                            "width": int(bbox[2]) if bbox is not None else None,
                            "height": int(bbox[3]) if bbox is not None else None,
                            "x2": int(bbox[0] + bbox[2]) if bbox is not None else None,
                            "y2": int(bbox[1] + bbox[3]) if bbox is not None else None
                        } if bbox is not None else None
                    })
            print("WAKTU: Waktu handle kecocokan untuk wajah ke-", i+1, ":", time.perf_counter() - timer_handle_match)
        print("WAKTU: Waktu proses semua wajah:", time.perf_counter() - timer_proses_wajah)
        print("WAKTU: Waktu total pengenalan wajah banyak:", time.perf_counter() - timer_awal)
        return results
    
    def handle_known_visitor(self, visitor_id, embedding, all_visitor_embeddings):
        added = self.database.maybe_add_visitor_embedding(visitor_id, embedding, all_visitor_embeddings)
        if added:
            # Refresh cache karena ada visitor embedding baru
            self.refresh_embeddings_cache()
        print("Info: Memperbarui embedding selesai...", visitor_id)
        print("info visitor_id:", visitor_id)
        
        visitor_id_obj = ObjectId(visitor_id)
        temp_person = visitor_collection.find_one({"_id": visitor_id_obj})
        print("debug: Data visitor ditemukan:", temp_person["name"])
        
        # Cek apakah sudah melewati 45 menit sejak last_seen
        last_seen = datetime.fromisoformat(temp_person["last_seen"])
        current_time = datetime.now()
        time_difference = current_time - last_seen
        
        # Increment appearance_count hanya jika sudah lebih dari 45 menit
        if time_difference.total_seconds() >= 45 * 60:  # 45 menit = 2700 detik
            temp_person["appearance_count"] += 1
            print(f"Info: Increment appearance_count karena sudah {time_difference.total_seconds() / 60:.1f} menit sejak last_seen")
        else:
            print(f"Info: Tidak increment appearance_count karena baru {time_difference.total_seconds() / 60:.1f} menit sejak last_seen")
        temp_person["last_seen"] = datetime.now().isoformat()
        
        print("debug: Memperbarui data visitor di database...")
        visitor_collection.update_one(
            {"_id": visitor_id_obj},
            {"$set": {
                "last_seen": temp_person["last_seen"],
                "appearance_count": temp_person["appearance_count"]
            }}
        )

        should_promote = (
            temp_person["appearance_count"] >= MIN_APPEARANCE_FOR_PROMOTION
        )
        if should_promote:
            # Promosi visitor ke user
            visitor_count = users_collection.count_documents({"name": {"$regex": r"^Visitor\d+$"}})
            new_visitor_name = f"Visitor{visitor_count + 1}"
            new_user = {
                "name": new_visitor_name,
                "created_at": datetime.now().isoformat()
            }
            user_result = users_collection.insert_one(new_user)
            user_id = user_result.inserted_id
            user_id = ObjectId(user_id)
            
            # Pindahkan semua embedding visitor ke vector_collection dengan user_id baru
            visitor_vectors = list(visitor_vector_collection.find({"visitor_id": visitor_id_obj}))
            for vec in visitor_vectors:
                new_vector = {
                    "user_id": user_id,
                    "embedding": vec["embedding"],
                    "created_at": vec["created_at"]
                }
                vector_collection.insert_one(new_vector)
            
            attendance_records = list(attendance_collection.find({"visitor_id": visitor_id_obj}))
            for record in attendance_records:
                attendance_collection.update_one(
                    {"_id": record["_id"]},
                    {
                        "$set": {"user_id": user_id},
                        "$unset": {"visitor_id": ""}
                    }
                )
            
            # Hapus data visitor
            visitor_vector_collection.delete_many({"visitor_id": visitor_id_obj})
            visitor_collection.delete_one({"_id": visitor_id_obj})
            
            # Refresh cache karena visitor dipromosikan ke user
            self.refresh_embeddings_cache()
            
            print(f"Info: Visitor {visitor_id} dipromosikan ke user dengan ID {user_id}.")
    
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
            
            # Step 3: Database Matching (gunakan cached embeddings)
            matching_start = time.perf_counter()
            user_id, distance = self.database.find_closest_match(embedding, threshold, self._cached_embeddings)
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
            class_id: ID kelas untuk absensi
            threshold: Threshold untuk recognition
            
        Returns:
            List of recognition results
        """
        image = self.load_image_from_base64(image_base64)
        if image is None:
            return []
        return self.recognize_faces(image, threshold)
    def recognize_from_base64_many(self, image_base64: str, class_id: str, threshold: float = None) -> List[Dict]:
        """
        Kenali wajah dari Base64 string
        
        Args:
            image_base64: String Base64 dari gambar
            threshold: Threshold untuk recognition
            
        Returns:
            List of recognition results
        """
        timerawal = time.perf_counter()
        image = self.load_image_from_base64(image_base64)
        print("Debug: Waktu load image dari base64:")
        timerstoptampilkantimer(timerawal)
        
        if image is None:
            return []
        return self.recognize_faces(image, class_id, threshold)
    
    def prepare_openset_dataset(self, registered_count: int = 20) -> Dict:
        """
        Split ChokePoint dataset menjadi registered users dan unknown people
        
        Args:
            registered_count: Jumlah orang untuk registered (default 20 dari 30)
            
        Returns:
            Dictionary dengan path ke registered_folder dan unknown_folder
        """
        print(f"\n{'='*60}")
        print("Mempersiapkan Open-Set Dataset")
        print(f"{'='*60}")
        
        chokepoint_dir = Path(CHOCK_POINT_IMAGES)
        
        if not chokepoint_dir.exists():
            raise ValueError(f"ChokePoint dataset tidak ditemukan: {chokepoint_dir}")
        
        # Get all person folders (0001-0030)
        person_folders = sorted([f for f in chokepoint_dir.iterdir() 
                               if f.is_dir() and f.name.isdigit()])
        
        if len(person_folders) < registered_count + 5:
            raise ValueError(f"Dataset terlalu kecil: {len(person_folders)} orang, "
                           f"minimal {registered_count + 5}")
        
        # Split: first N for registered, rest for unknown
        registered_folders = person_folders[:registered_count]
        unknown_folders = person_folders[registered_count:]
        
        # Create output directories
        registered_dir = Path(REGISTERED_USERS_DIR)
        unknown_dir = Path(UNKNOWN_PEOPLE_DIR)
        
        # Clear existing data
        if registered_dir.exists():
            shutil.rmtree(registered_dir)
        if unknown_dir.exists():
            shutil.rmtree(unknown_dir)
        
        registered_dir.mkdir(parents=True, exist_ok=True)
        unknown_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        registered_count_images = 0
        unknown_count_images = 0
        
        print(f"\nMenyalin {len(registered_folders)} orang ke registered_users/...")
        for folder in tqdm(registered_folders, desc="Registered"):
            for img_file in folder.iterdir():
                if img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    dest = registered_dir / img_file.name
                    shutil.copy2(img_file, dest)
                    registered_count_images += 1
        
        print(f"Menyalin {len(unknown_folders)} orang ke unknown_people/...")
        for folder in tqdm(unknown_folders, desc="Unknown"):
            for img_file in folder.iterdir():
                if img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    dest = unknown_dir / img_file.name
                    shutil.copy2(img_file, dest)
                    unknown_count_images += 1
        
        result = {
            "registered_folder": str(registered_dir),
            "unknown_folder": str(unknown_dir),
            "registered_persons": len(registered_folders),
            "registered_images": registered_count_images,
            "unknown_persons": len(unknown_folders),
            "unknown_images": unknown_count_images,
            "registered_ids": [f.name for f in registered_folders],
            "unknown_ids": [f.name for f in unknown_folders]
        }
        
        print(f"\n{'='*60}")
        print("Dataset Open-Set Siap!")
        print(f"{'='*60}")
        print(f"Registered: {result['registered_persons']} orang, "
              f"{result['registered_images']} gambar")
        print(f"Unknown: {result['unknown_persons']} orang, "
              f"{result['unknown_images']} gambar")
        print(f"\nRegistered IDs: {', '.join(result['registered_ids'][:5])}...")
        print(f"Unknown IDs: {', '.join(result['unknown_ids'][:5])}...")
        
        return result
    
    def _evaluate_single_threshold(self, 
                                   registered_images: List[Tuple[str, str]],
                                   unknown_images: List[Tuple[str, str]],
                                   threshold: float) -> Dict:
        """
        Evaluasi open-set recognition untuk satu threshold
        
        Args:
            registered_images: List of (image_path, true_name) untuk registered users
            unknown_images: List of (image_path, true_name) untuk unknown people
            threshold: Threshold untuk recognition
            
        Returns:
            Dictionary dengan TP, FP, FN, TN counts dan detail errors
        """
        # Metrics counters
        true_positives = 0  # Registered user dikenali benar
        false_positives = 0  # Registered user dikenali salah (sebagai user lain)
        false_negatives = 0  # Registered user tidak dikenali (jadi visitor)
        true_negatives = 0  # Unknown ditolak (jadi visitor)
        false_accepts = 0  # Unknown diterima sebagai registered user (CRITICAL ERROR)
        
        details = {
            "correct_identifications": [],
            "misidentifications": [],
            "false_rejections": [],
            "correct_rejections": [],
            "false_accepts": [],
            "no_face_detected": []
        }
        
        # Test registered users
        for img_path, true_name in tqdm(registered_images, desc=f"Testing Registered (t={threshold})", leave=False):
            image = self.load_image(img_path)
            if image is None:
                details["no_face_detected"].append({"file": img_path, "category": "registered"})
                false_negatives += 1
                continue
            
            face = self.detector.detect_single_face(image)
            if face is None:
                details["no_face_detected"].append({"file": img_path, "category": "registered"})
                false_negatives += 1
                continue
            
            embedding = self.encoder.get_embedding(face)
            if embedding is None:
                details["no_face_detected"].append({"file": img_path, "category": "registered"})
                false_negatives += 1
                continue
            
            # Find match (gunakan cached embeddings)
            user_id, distance = self.database.find_closest_match(embedding, self._cached_embeddings, threshold)
            
            if user_id is None:
                # Registered user ditolak → False Negative (False Rejection)
                false_negatives += 1
                details["false_rejections"].append({
                    "file": Path(img_path).name,
                    "true_name": true_name,
                    "predicted": "Visitor",
                    "distance": distance
                })
            else:
                # Get predicted name
                if isinstance(user_id, str):
                    user_id = ObjectId(user_id)
                user_doc = users_collection.find_one({"_id": user_id})
                predicted_name = user_doc.get("name", "Unknown") if user_doc else "Unknown"
                
                if predicted_name == true_name:
                    # Correct identification → True Positive
                    true_positives += 1
                    details["correct_identifications"].append({
                        "file": Path(img_path).name,
                        "name": true_name,
                        "distance": distance
                    })
                else:
                    # Misidentification (identified as different registered user) → False Positive
                    false_positives += 1
                    details["misidentifications"].append({
                        "file": Path(img_path).name,
                        "true_name": true_name,
                        "predicted": predicted_name,
                        "distance": distance
                    })
        
        # Test unknown people
        for img_path, true_name in tqdm(unknown_images, desc=f"Testing Unknown (t={threshold})", leave=False):
            image = self.load_image(img_path)
            if image is None:
                details["no_face_detected"].append({"file": img_path, "category": "unknown"})
                true_negatives += 1  # No detection = rejection = correct for unknown
                continue
            
            face = self.detector.detect_single_face(image)
            if face is None:
                details["no_face_detected"].append({"file": img_path, "category": "unknown"})
                true_negatives += 1
                continue
            
            embedding = self.encoder.get_embedding(face)
            if embedding is None:
                details["no_face_detected"].append({"file": img_path, "category": "unknown"})
                true_negatives += 1
                continue
            
            # Find match (gunakan cached embeddings)
            user_id, distance = self.database.find_closest_match(embedding, self._cached_embeddings, threshold)
            
            if user_id is None:
                # Unknown ditolak (jadi visitor) → True Negative (Correct Rejection)
                true_negatives += 1
                details["correct_rejections"].append({
                    "file": Path(img_path).name,
                    "true_name": true_name,
                    "distance": distance
                })
            else:
                # Unknown diterima sebagai registered user → False Accept (CRITICAL!)
                if isinstance(user_id, str):
                    user_id = ObjectId(user_id)
                user_doc = users_collection.find_one({"_id": user_id})
                predicted_name = user_doc.get("name", "Unknown") if user_doc else "Unknown"
                
                false_accepts += 1
                details["false_accepts"].append({
                    "file": Path(img_path).name,
                    "true_name": true_name,
                    "predicted": predicted_name,
                    "distance": distance
                })
        
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "false_accepts": false_accepts,
            "details": details
        }
    
    def test_open_set_accuracy(self,
                              registered_folder: str = None,
                              unknown_folder: str = None,
                              thresholds: List[float] = None,
                              threshold: float = None) -> Dict:
        """
        Test open-set recognition accuracy dengan metrik TAR, FAR, FRR, TRR
        
        Args:
            registered_folder: Folder berisi gambar registered users
            unknown_folder: Folder berisi gambar unknown people
            thresholds: List threshold untuk ROC curve (default: DEFAULT_THRESHOLDS)
            threshold: Single threshold untuk test cepat (override thresholds)
            
        Returns:
            Dictionary dengan metrik open-set evaluation:
            - per_threshold_results: Hasil untuk setiap threshold
            - roc_data: Data untuk plotting ROC curve
            - best_threshold: Threshold optimal (berdasarkan EER)
            - summary: Ringkasan hasil
        """
        print(f"\n{'='*60}")
        print("OPEN-SET FACE RECOGNITION EVALUATION")
        print(f"{'='*60}")
        
        # Use default folders if not provided
        if registered_folder is None:
            registered_folder = REGISTERED_USERS_DIR
        if unknown_folder is None:
            unknown_folder = UNKNOWN_PEOPLE_DIR
        
        reg_folder = Path(registered_folder)
        unk_folder = Path(unknown_folder)
        
        # Check folders exist
        if not reg_folder.exists():
            raise ValueError(f"Registered folder tidak ditemukan: {reg_folder}\n"
                           f"Jalankan prepare_openset_dataset() terlebih dahulu!")
        if not unk_folder.exists():
            raise ValueError(f"Unknown folder tidak ditemukan: {unk_folder}\n"
                           f"Jalankan prepare_openset_dataset() terlebih dahulu!")
        
        # Load images
        registered_images = []
        for img_file in reg_folder.iterdir():
            if img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                true_name = self.extract_name_from_filename(img_file.name)
                registered_images.append((str(img_file), true_name))
        
        unknown_images = []
        for img_file in unk_folder.iterdir():
            if img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                true_name = self.extract_name_from_filename(img_file.name)
                unknown_images.append((str(img_file), true_name))
        
        if not registered_images:
            raise ValueError(f"Tidak ada gambar registered ditemukan di {reg_folder}")
        if not unknown_images:
            raise ValueError(f"Tidak ada gambar unknown ditemukan di {unk_folder}")
        
        print(f"\nDataset:")
        print(f"  Registered users: {len(registered_images)} gambar")
        print(f"  Unknown people: {len(unknown_images)} gambar")
        
        # Determine thresholds to test
        if threshold is not None:
            # Single threshold mode
            test_thresholds = [threshold]
            print(f"\nMode: Single threshold testing ({threshold})")
        else:
            # Multi-threshold mode for ROC curve
            test_thresholds = thresholds if thresholds else DEFAULT_THRESHOLDS
            print(f"\nMode: Multi-threshold testing ({len(test_thresholds)} thresholds)")
        
        # Evaluate for each threshold
        results_per_threshold = []
        
        print(f"\n{'='*60}")
        print("Memulai Evaluasi...")
        print(f"{'='*60}")
        
        for t in test_thresholds:
            print(f"\n[Threshold {t:.2f}]")
            start_time = time.perf_counter()
            
            result = self._evaluate_single_threshold(registered_images, unknown_images, t)
            
            elapsed = time.perf_counter() - start_time
            
            # Calculate metrics
            tp = result["true_positives"]
            fp = result["false_positives"]
            fn = result["false_negatives"]
            tn = result["true_negatives"]
            fa = result["false_accepts"]
            
            # TAR (True Acceptance Rate) = TP / (TP + FN)
            # Percentage of registered users correctly identified
            tar = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            
            # FRR (False Rejection Rate) = FN / (TP + FN)
            # Percentage of registered users incorrectly rejected (as visitor)
            frr = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0
            
            # FAR (False Acceptance Rate) = FA / (FA + TN)
            # Percentage of unknown people incorrectly accepted (as registered user)
            far = (fa / (fa + tn) * 100) if (fa + tn) > 0 else 0
            
            # TRR (True Rejection Rate) = TN / (FA + TN)
            # Percentage of unknown people correctly rejected (as visitor)
            trr = (tn / (fa + tn) * 100) if (fa + tn) > 0 else 0
            
            # Additional metrics
            # Accuracy for registered users only
            registered_accuracy = (tp / (tp + fp + fn) * 100) if (tp + fp + fn) > 0 else 0
            
            # Overall accuracy (both registered and unknown)
            total = tp + fp + fn + tn + fa
            overall_accuracy = ((tp + tn) / total * 100) if total > 0 else 0
            
            threshold_result = {
                "threshold": t,
                "confusion_matrix": {
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_negatives": tn,
                    "false_accepts": fa
                },
                "metrics": {
                    "tar": tar,  # True Acceptance Rate
                    "frr": frr,  # False Rejection Rate
                    "far": far,  # False Acceptance Rate (CRITICAL METRIC)
                    "trr": trr,  # True Rejection Rate
                    "registered_accuracy": registered_accuracy,
                    "overall_accuracy": overall_accuracy
                },
                "evaluation_time_seconds": elapsed,
                "details": result["details"]
            }
            
            results_per_threshold.append(threshold_result)
            
            print(f"  TAR: {tar:.2f}%  |  FRR: {frr:.2f}%")
            print(f"  FAR: {far:.2f}%  |  TRR: {trr:.2f}%")
            print(f"  Overall Accuracy: {overall_accuracy:.2f}%")
            print(f"  Time: {elapsed:.2f}s")
        
        # Calculate ROC curve data (FPR vs TPR)
        roc_data = []
        for result in results_per_threshold:
            # TPR = TAR (True Positive Rate)
            tpr = result["metrics"]["tar"]
            # FPR = FAR (False Positive Rate)
            fpr = result["metrics"]["far"]
            
            roc_data.append({
                "threshold": result["threshold"],
                "fpr": fpr,
                "tpr": tpr,
                "frr": result["metrics"]["frr"]
            })
        
        # Find best threshold (closest to EER: Equal Error Rate where FAR ≈ FRR)
        best_threshold_idx = 0
        min_eer_diff = float('inf')
        for i, result in enumerate(results_per_threshold):
            far = result["metrics"]["far"]
            frr = result["metrics"]["frr"]
            eer_diff = abs(far - frr)
            if eer_diff < min_eer_diff:
                min_eer_diff = eer_diff
                best_threshold_idx = i
        
        best_result = results_per_threshold[best_threshold_idx]
        
        # Summary
        summary = {
            "total_registered_images": len(registered_images),
            "total_unknown_images": len(unknown_images),
            "registered_persons": len(set([name for _, name in registered_images])),
            "unknown_persons": len(set([name for _, name in unknown_images])),
            "thresholds_tested": len(test_thresholds),
            "best_threshold": best_result["threshold"],
            "best_metrics": best_result["metrics"],
            "eer_approximate": (best_result["metrics"]["far"] + best_result["metrics"]["frr"]) / 2
        }
        
        # Print final summary
        print(f"\n{'='*60}")
        print("HASIL EVALUASI OPEN-SET")
        print(f"{'='*60}")
        print(f"\nBest Threshold: {summary['best_threshold']:.2f}")
        print(f"  TAR (True Accept): {summary['best_metrics']['tar']:.2f}%")
        print(f"  FAR (False Accept): {summary['best_metrics']['far']:.2f}%  ← CRITICAL")
        print(f"  FRR (False Reject): {summary['best_metrics']['frr']:.2f}%")
        print(f"  TRR (True Reject): {summary['best_metrics']['trr']:.2f}%")
        print(f"  Overall Accuracy: {summary['best_metrics']['overall_accuracy']:.2f}%")
        print(f"  EER ≈ {summary['eer_approximate']:.2f}%")
        
        # Check against security requirements
        print(f"\n{'='*60}")
        print("EVALUASI KEAMANAN")
        print(f"{'='*60}")
        far_pass = summary['best_metrics']['far'] < 1.0
        tar_pass = summary['best_metrics']['tar'] >= 95.0
        print(f"FAR < 1%: {'✓ PASS' if far_pass else '✗ FAIL'} "
              f"({summary['best_metrics']['far']:.2f}%)")
        print(f"TAR >= 95%: {'✓ PASS' if tar_pass else '✗ FAIL'} "
              f"({summary['best_metrics']['tar']:.2f}%)")
        
        return {
            "per_threshold_results": results_per_threshold,
            "roc_data": roc_data,
            "best_threshold": summary["best_threshold"],
            "summary": summary
        }
    
    def test_open_set_with_groundtruth(self,
                                       ground_truth_file: str,
                                       thresholds: List[float] = None,
                                       threshold: float = None) -> Dict:
        """
        Test open-set accuracy dengan ground truth manual untuk CCTV frames
        Cocok untuk testing dengan real CCTV frames + unknown people dari LFW
        
        Args:
            ground_truth_file: Path ke JSON file berisi ground truth
            thresholds: List threshold untuk ROC curve (default dari config)
            threshold: Single threshold untuk quick test (override thresholds)
            
        Returns:
            Dict berisi TAR, FAR, FRR, TRR, ROC data, best threshold
        """
        print(f"\n{'='*60}")
        print("OPEN-SET EVALUATION WITH GROUND TRUTH")
        print(f"{'='*60}")
        
        # Determine thresholds
        if thresholds is None and threshold is None:
            thresholds = DEFAULT_THRESHOLDS
        elif threshold is not None:
            thresholds = [threshold]
        
        # Load ground truth
        gt_path = Path(ground_truth_file)
        if not gt_path.exists():
            raise ValueError(f"Ground truth file tidak ditemukan: {ground_truth_file}")
        
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        registered_tests = ground_truth.get("registered_tests", [])
        unknown_tests = ground_truth.get("unknown_tests", [])
        
        if not registered_tests:
            raise ValueError("Ground truth file tidak memiliki 'registered_tests'")
        if not unknown_tests:
            raise ValueError("Ground truth file tidak memiliki 'unknown_tests'")
        
        print(f"\nDataset:")
        print(f"  Registered user tests: {len(registered_tests)} images")
        print(f"  Unknown people tests: {len(unknown_tests)} images")
        print(f"  Total ground truth IDs in registered: {sum(len(t.get('ground_truth_ids', [])) for t in registered_tests)}")
        print(f"\nThresholds to test: {thresholds}")
        
        # Evaluate for each threshold
        results_per_threshold = []
        
        print(f"\n{'='*60}")
        print("Memulai Evaluasi...")
        print(f"{'='*60}")
        
        for thresh in thresholds:
            print(f"\n[Threshold {thresh:.2f}]")
            start_time = time.perf_counter()
            
            result = self._evaluate_with_groundtruth(
                registered_tests,
                unknown_tests,
                thresh
            )
            
            elapsed = time.perf_counter() - start_time
            result['evaluation_time_seconds'] = elapsed
            
            results_per_threshold.append(result)
            
            print(f"  TAR: {result['tar']:.2f}%  |  FRR: {result['frr']:.2f}%")
            print(f"  FAR: {result['far']:.2f}%  |  TRR: {result['trr']:.2f}%")
            print(f"  Overall Accuracy: {result['overall_accuracy']:.2f}%")
            print(f"  Time: {elapsed:.2f}s")
        
        # Find best threshold (highest balanced accuracy)
        best_idx = max(range(len(results_per_threshold)),
                      key=lambda i: results_per_threshold[i]['balanced_accuracy'])
        best_result = results_per_threshold[best_idx]
        
        # Calculate EER approximation (where FAR ≈ FRR)
        eer_approx = None
        eer_threshold = None
        min_diff = float('inf')
        for result in results_per_threshold:
            diff = abs(result['far'] - result['frr'])
            if diff < min_diff:
                min_diff = diff
                eer_approx = (result['far'] + result['frr']) / 2
                eer_threshold = result['threshold']
        
        # Prepare ROC data
        roc_data = [{
            'threshold': r['threshold'],
            'fpr': r['far'],
            'tpr': r['tar'],
            'far': r['far'],
            'frr': r['frr']
        } for r in results_per_threshold]
        
        # Summary
        summary = {
            'test_type': 'open_set_with_groundtruth',
            'dataset_info': {
                'registered_test_images': len(registered_tests),
                'unknown_test_images': len(unknown_tests),
                'total_registered_faces': sum(len(t.get('ground_truth_ids', [])) for t in registered_tests),
                'total_tests': len(registered_tests) + len(unknown_tests)
            },
            'thresholds_tested': len(thresholds),
            'best_threshold': best_result['threshold'],
            'best_metrics': {
                'tar': best_result['tar'],
                'far': best_result['far'],
                'frr': best_result['frr'],
                'trr': best_result['trr'],
                'balanced_accuracy': best_result['balanced_accuracy'],
                'overall_accuracy': best_result['overall_accuracy']
            },
            'eer_info': {
                'eer_approximate': eer_approx,
                'eer_threshold': eer_threshold
            }
        }
        
        # Print final summary
        print(f"\n{'='*60}")
        print("HASIL EVALUASI OPEN-SET (GROUND TRUTH)")
        print(f"{'='*60}")
        print(f"\nBest Threshold (Balanced Accuracy): {summary['best_threshold']:.2f}")
        print(f"  TAR (True Accept): {summary['best_metrics']['tar']:.2f}%")
        print(f"  FAR (False Accept): {summary['best_metrics']['far']:.2f}%  ← CRITICAL")
        print(f"  FRR (False Reject): {summary['best_metrics']['frr']:.2f}%")
        print(f"  TRR (True Reject): {summary['best_metrics']['trr']:.2f}%")
        print(f"  Balanced Accuracy: {summary['best_metrics']['balanced_accuracy']:.2f}%")
        print(f"  Overall Accuracy: {summary['best_metrics']['overall_accuracy']:.2f}%")
        
        if eer_approx:
            print(f"\nEER (Equal Error Rate): {eer_approx:.2f}% @ threshold {eer_threshold:.2f}")
        
        # Check against security requirements
        print(f"\n{'='*60}")
        print("EVALUASI KEAMANAN")
        print(f"{'='*60}")
        far_pass = summary['best_metrics']['far'] < 1.0
        tar_pass = summary['best_metrics']['tar'] >= 95.0
        print(f"FAR < 1%: {'✓ PASS' if far_pass else '✗ FAIL'} "
              f"({summary['best_metrics']['far']:.2f}%)")
        print(f"TAR >= 95%: {'✓ PASS' if tar_pass else '✗ FAIL'} "
              f"({summary['best_metrics']['tar']:.2f}%)")
        
        return {
            'per_threshold_results': results_per_threshold,
            'roc_data': roc_data,
            'best_threshold': summary['best_threshold'],
            'summary': summary
        }
    
    def _evaluate_with_groundtruth(self,
                                   registered_tests: List[Dict],
                                   unknown_tests: List[Dict],
                                   threshold: float) -> Dict:
        """
        Evaluasi single threshold dengan ground truth manual
        
        Args:
            registered_tests: List dict dengan 'image' dan 'ground_truth_ids'
            unknown_tests: List dict dengan 'image' (unknown people)
            threshold: Threshold value
            
        Returns:
            Dict dengan metrics untuk threshold ini
        """
        # Counters
        tp = 0  # Registered correctly recognized
        fp = 0  # Unknown wrongly accepted as registered
        fn = 0  # Registered wrongly rejected (jadi visitor)
        tn = 0  # Unknown correctly rejected (jadi visitor)
        
        registered_errors = []
        unknown_errors = []
        no_detection_count = 0
        
        # Test registered users (should be recognized)
        for test in tqdm(registered_tests, desc=f"Testing registered @ {threshold:.2f}", leave=False):
            image_path = test['image']
            ground_truth_ids = test.get('ground_truth_ids', [])
            
            if not ground_truth_ids:
                continue
            
            # Load and recognize
            image = self.load_image(image_path)
            if image is None:
                fn += len(ground_truth_ids)
                no_detection_count += 1
                continue
            
            # Recognize all faces in image (using temporary class_id for testing)
            try:
                results = self.recognize_faces(image, class_id="eval_test", threshold=threshold)
            except Exception as e:
                print(f"Error recognizing {image_path}: {e}")
                fn += len(ground_truth_ids)
                continue
            
            # Get recognized user IDs
            recognized_user_ids = []
            for r in results:
                if 'user_id' in r:
                    # Get name from database
                    user_id_obj = ObjectId(r['user_id']) if isinstance(r['user_id'], str) else r['user_id']
                    user_doc = users_collection.find_one({"_id": user_id_obj})
                    if user_doc:
                        recognized_user_ids.append(user_doc.get('name'))
            
            # Check each ground truth ID
            for gt_id in ground_truth_ids:
                if gt_id in recognized_user_ids:
                    tp += 1
                else:
                    fn += 1
                    registered_errors.append({
                        'image': Path(image_path).name,
                        'expected': gt_id,
                        'got': recognized_user_ids if recognized_user_ids else 'visitor/no_detection'
                    })
        
        # Test unknown people (should be rejected as visitor)
        for test in tqdm(unknown_tests, desc=f"Testing unknown @ {threshold:.2f}", leave=False):
            image_path = test['image']
            
            # Load and recognize
            image = self.load_image(image_path)
            if image is None:
                tn += 1  # No detection = rejection = correct for unknown
                continue
            
            # Recognize
            try:
                results = self.recognize_faces(image, class_id="eval_test", threshold=threshold)
            except Exception as e:
                print(f"Error recognizing {image_path}: {e}")
                tn += 1
                continue
            
            # Check if any face was wrongly accepted as registered user
            if results:
                has_registered = False
                wrongly_accepted = []
                
                for r in results:
                    if 'user_id' in r:
                        has_registered = True
                        # Get name
                        user_id_obj = ObjectId(r['user_id']) if isinstance(r['user_id'], str) else r['user_id']
                        user_doc = users_collection.find_one({"_id": user_id_obj})
                        if user_doc:
                            wrongly_accepted.append(user_doc.get('name'))
                
                if has_registered:
                    fp += 1  # False Accept (CRITICAL ERROR)
                    unknown_errors.append({
                        'image': Path(image_path).name,
                        'wrongly_accepted_as': wrongly_accepted
                    })
                else:
                    tn += 1  # Correctly rejected (all faces became visitors)
            else:
                tn += 1  # No detection = rejection = correct
        
        # Calculate metrics
        tar = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        far = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
        frr = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0
        trr = (tn / (fp + tn) * 100) if (fp + tn) > 0 else 0
        
        total = tp + fp + fn + tn
        overall_accuracy = ((tp + tn) / total * 100) if total > 0 else 0
        balanced_accuracy = (tar + trr) / 2
        
        return {
            'threshold': threshold,
            'confusion_matrix': {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            },
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'tar': tar,
            'far': far,
            'frr': frr,
            'trr': trr,
            'overall_accuracy': overall_accuracy,
            'balanced_accuracy': balanced_accuracy,
            'no_detection_count': no_detection_count,
            'errors': {
                'registered_errors': registered_errors[:10],  # Max 10 examples
                'unknown_errors': unknown_errors[:10]
            }
        }
    
    
    def close(self):
        """
        Tutup sistem
        """
        self.database.close()
        