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
    
    def register_faces_from_folder(self, folder_path: str = None, clear_existing: bool = True, dataset = DATABASE_IMAGES_DIR) -> Dict:
        """
        Daftarkan semua wajah dari folder ke database
        
        Args:
            folder_path: Path ke folder berisi gambar wajah
            clear_existing: Hapus data existing sebelum registrasi
            
        Returns:
            Dictionary dengan statistik registrasi
        """
        if folder_path is None:
            folder_path = dataset
        
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
    
    def test_open_set_with_groundtruth(self,
                                       ground_truth_file: str,
                                       threshold: float = None) -> Dict:
        """
        Test open-set accuracy dengan ground truth manual untuk CCTV frames
        Cocok untuk testing dengan real CCTV frames + unknown people dari LFW
        
        Args:
            ground_truth_file: Path ke JSON file berisi ground truth
            threshold: Threshold untuk recognition (default dari RECOGNITION_THRESHOLD)
            
        Returns:
            Dict berisi TAR, FAR, FRR, TRR dan metrics lainnya
        """
        print(f"\n{'='*60}")
        print("OPEN-SET EVALUATION WITH GROUND TRUTH")
        print(f"{'='*60}")
        
        # Use default threshold if not specified
        if threshold is None:
            threshold = RECOGNITION_THRESHOLD
        
        print(f"\nThreshold: {threshold}")
        
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
        
        # Evaluate with single threshold
        print(f"\n{'='*60}")
        print("Memulai Evaluasi...")
        print(f"{'='*60}")
        
        start_time = time.perf_counter()
        
        result = self._evaluate_with_groundtruth(
            registered_tests,
            unknown_tests,
            threshold
        )
        
        elapsed = time.perf_counter() - start_time
        result['evaluation_time_seconds'] = elapsed
        
        # Print results
        print(f"\n{'='*60}")
        print("HASIL EVALUASI")
        print(f"{'='*60}")
        print(f"  TAR (True Accept): {result['tar']:.2f}%")
        print(f"  FAR (False Accept): {result['far']:.2f}%  ← CRITICAL")
        print(f"  FRR (False Reject): {result['frr']:.2f}%")
        print(f"  TRR (True Reject): {result['trr']:.2f}%")
        print(f"  Balanced Accuracy: {result['balanced_accuracy']:.2f}%")
        print(f"  Overall Accuracy: {result['overall_accuracy']:.2f}%")
        print(f"  Evaluation Time: {elapsed:.2f}s")
        
        # Check against security requirements
        print(f"\n{'='*60}")
        print("EVALUASI KEAMANAN")
        print(f"{'='*60}")
        far_pass = result['far'] < 1.0
        tar_pass = result['tar'] >= 95.0
        print(f"FAR < 1%: {'✓ PASS' if far_pass else '✗ FAIL'} ({result['far']:.2f}%)")
        print(f"TAR >= 95%: {'✓ PASS' if tar_pass else '✗ FAIL'} ({result['tar']:.2f}%)")
        
        # Summary
        summary = {
            'test_type': 'open_set_with_groundtruth',
            'threshold': threshold,
            'dataset_info': {
                'registered_test_images': len(registered_tests),
                'unknown_test_images': len(unknown_tests),
                'total_registered_faces': sum(len(t.get('ground_truth_ids', [])) for t in registered_tests),
                'total_tests': len(registered_tests) + len(unknown_tests)
            },
            'metrics': {
                'tar': result['tar'],
                'far': result['far'],
                'frr': result['frr'],
                'trr': result['trr'],
                'balanced_accuracy': result['balanced_accuracy'],
                'overall_accuracy': result['overall_accuracy']
            },
            'confusion_matrix': result['confusion_matrix'],
            'evaluation_time_seconds': elapsed,
            'security_check': {
                'far_pass': far_pass,
                'tar_pass': tar_pass
            }
        }
        
        return {
            'result': result,
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
                results = self.recognize_faces(image, class_id="694f3f07825e7bcbbe801d10", threshold=threshold)
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
                results = self.recognize_faces(image, class_id="694f3f07825e7bcbbe801d10", threshold=threshold)
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
        