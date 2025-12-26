"""
Database handler menggunakan MongoDB untuk menyimpan face embeddings
"""
from .face_encoder import FaceEncoder
import numpy as np
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME
from config.configrations import attendance_collection, users_collection, vector_collection, visitor_vector_collection, visitor_collection


class FaceDatabase:
    """
    Handler untuk menyimpan dan mengambil face embeddings dari MongoDB
    """
    
    def __init__(self):
        """
        Inisialisasi koneksi MongoDB
        """
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            print(f"Terhubung ke MongoDB: {MONGO_URI}")
            
            self.db = self.client[MONGO_DB_NAME]
            self.collection = self.db[MONGO_COLLECTION_NAME]
            
            # Create index untuk pencarian cepat
            self.collection.create_index("person_name")
            
        except ConnectionFailure as e:
            print(f"Gagal terhubung ke MongoDB: {e}")
            print("Menggunakan mode in-memory sebagai fallback...")
            self.client = None
            self.db = None
            self.collection = None
            self._memory_storage: List[Dict] = []
    
    def get_all_embeddings(self) -> List[Dict]:
        """
        Ambil semua embeddings dari database
        
        Returns:
            List of documents dengan embedding
        """
        if vector_collection is not None:
            documents = list(vector_collection.find({}, {"_id": 0}))
        else:
            documents = self._memory_storage
        
        # Convert embedding list back ke numpy array dan bersihkan ObjectId
        cleaned_documents = []
        for doc in documents:
            cleaned_doc = {}
            for key, value in doc.items():
                if key == "_id":
                    continue  # Skip _id
                elif isinstance(value, list) and key == "embedding":
                    cleaned_doc[key] = np.array(value)
                elif isinstance(value, ObjectId):
                    cleaned_doc[key] = str(value)
                else:
                    cleaned_doc[key] = value
            cleaned_documents.append(cleaned_doc)
        
        return cleaned_documents
    
    def get_all_visitor_embeddings(self) -> List[Dict]:
        """
        Ambil semua embeddings dari database
        
        Returns:
            List of documents dengan embedding
        """
        if visitor_vector_collection is not None:
            documents = list(visitor_vector_collection.find({}, {"_id": 0}))
        else:
            documents = self._memory_storage
        
        # Convert embedding list back ke numpy array dan bersihkan ObjectId
        cleaned_documents = []
        for doc in documents:
            cleaned_doc = {}
            for key, value in doc.items():
                if key == "_id":
                    continue  # Skip _id
                elif isinstance(value, list) and key == "embedding":
                    cleaned_doc[key] = np.array(value)
                elif isinstance(value, ObjectId):
                    cleaned_doc[key] = str(value)
                else:
                    cleaned_doc[key] = value
            cleaned_documents.append(cleaned_doc)
        
        return cleaned_documents
    
    def get_embeddings_by_person(self, person_name: str) -> List[np.ndarray]:
        """
        Ambil semua embeddings untuk satu orang
        
        Args:
            person_name: Nama orang
            
        Returns:
            List of embeddings
        """
        if self.collection is not None:
            documents = list(self.collection.find({"person_name": person_name}))
        else:
            documents = [d for d in self._memory_storage if d["person_name"] == person_name]
        
        return [np.array(doc["embedding"]) for doc in documents]
    
    def get_unique_persons(self) -> List[str]:
        """
        Ambil daftar nama unik dari database
        
        Returns:
            List of unique person names
        """
        if self.collection is not None:
            return self.collection.distinct("person_name")
        else:
            return list(set(d["person_name"] for d in self._memory_storage))
    
    def get_person_count(self) -> int:
        """
        Hitung jumlah orang unik dalam database
        
        Returns:
            Jumlah orang
        """
        return len(self.get_unique_persons())
    
    def get_embedding_count(self) -> int:
        """
        Hitung total embeddings dalam database
        
        Returns:
            Jumlah embeddings
        """
        if self.collection is not None:
            return self.collection.count_documents({})
        else:
            return len(self._memory_storage)
    
    def delete_person(self, person_name: str) -> int:
        """
        Hapus semua embeddings untuk satu orang
        
        Args:
            person_name: Nama orang
            
        Returns:
            Jumlah dokumen yang dihapus
        """
        if self.collection is not None:
            result = self.collection.delete_many({"person_name": person_name})
            return result.deleted_count
        else:
            before = len(self._memory_storage)
            self._memory_storage = [d for d in self._memory_storage if d["person_name"] != person_name]
            return before - len(self._memory_storage)
    
    def clear_database(self) -> int:
        """
        Hapus semua data dari database
        
        Returns:
            Jumlah dokumen yang dihapus
        """
        if self.collection is not None:
            result = self.collection.delete_many({})
            return result.deleted_count
        else:
            count = len(self._memory_storage)
            self._memory_storage = []
            return count
    
    def find_closest_match(self, query_embedding: np.ndarray, threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Cari embedding terdekat menggunakan Euclidean distance
        
        Args:
            query_embedding: Embedding yang akan dicari
            threshold: Maximum distance untuk dianggap match
            
        Returns:
            Tuple (person_name, distance) atau (None, float('inf')) jika tidak ada match
        """
        all_embeddings = self.get_all_embeddings()
    
        if not all_embeddings:
            return None, 0.0
        
        max_similarity = -1.0  # ← Mulai dari -1 (untuk cari MAXIMUM)
        best_match = None
        
        for doc in all_embeddings:
            similarity = FaceEncoder.compute_cosine_similarity(
                query_embedding, 
                doc["embedding"]
            )
            
            if similarity > max_similarity: 
                max_similarity = similarity
                best_match = doc["user_id"]
        
        # nama_prediksi = users_collection.find_one({"_id": ObjectId(best_match)})["name"] if best_match else None
        # print(f"Debug: Closest match: {best_match} ({nama_prediksi}) dengan similarity {max_similarity}")
        if max_similarity >= threshold:
            return best_match, max_similarity
        else:
            return None, max_similarity
        
    def find_visitor_closest_match(self, query_embedding: np.ndarray, threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Cari embedding terdekat menggunakan Euclidean distance
        
        Args:
            query_embedding: Embedding yang akan dicari
            threshold: Maximum distance untuk dianggap match
            
        Returns:
            Tuple (person_name, distance) atau (None, float('inf')) jika tidak ada match
        """
        all_embeddings = self.get_all_visitor_embeddings()
        print(f"Debug: Mencari di database visitor, total embeddings: {len(all_embeddings)}")
        if not all_embeddings:
            return None, 0.0
        
        max_similarity = -1.0  # ← Mulai dari -1 (untuk cari MAXIMUM)
        best_match = None
        
        for doc in all_embeddings:
            similarity = FaceEncoder.compute_cosine_similarity(
                query_embedding, 
                doc["embedding"]
            )
            
            if similarity > max_similarity: 
                max_similarity = similarity
                best_match = doc["visitor_id"]
        
        print(f"Debug: Closest visitor match: {best_match} dengan similarity {max_similarity}")
        if max_similarity >= threshold:
            return best_match, max_similarity
        else:
            return None, max_similarity
    
    def maybe_add_embedding(self, user_id, embedding):
        """
        Tambah embedding baru jika user belum memiliki embedding,
        atau jika embedding yang ada sudah berbeda signifikan

        Args:
            user_id: ID user
            embedding: Face embedding numpy array

        Returns:
            bool: True jika embedding ditambahkan, False jika tidak
        """
        if vector_collection is not None:
            existing_embeddings = list(vector_collection.find({"user_id": user_id}))
            count = len(existing_embeddings)
            user_id = ObjectId(user_id)
            
            if count <= 10:
                # Cek similarity dengan embedding yang sudah ada
                similarity = self.find_closest_match(embedding)[1]
                
                # Jika similarity > 0.85, embedding terlalu mirip, jangan tambah
                if similarity > 0.85:
                    return False
                
                document = {
                    "user_id": user_id,
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "created_at": datetime.now()
                }
                vector_collection.insert_one(document)
                return True
            else:
                return False
        else:
            # Fallback ke memory storage
            existing = [d for d in self._memory_storage if d.get("user_id") == user_id]
            count = len(existing)
            
            if count == 0:
                document = {
                    "user_id": user_id,
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "created_at": datetime.now()
                }
                self._memory_storage.append(document)
                return True
            else:
                return False

    def maybe_add_visitor_embedding(self, visitor_id, embedding):
        """
        Tambah embedding baru jika visitor belum memiliki embedding,
        atau jika embedding yang ada sudah berbeda signifikan

        Args:
            visitor_id: ID visitor
            embedding: Face embedding numpy array

        Returns:
            bool: True jika embedding ditambahkan, False jika tidak
        """
        if visitor_vector_collection is not None:
            existing_embeddings = list(visitor_vector_collection.find({"visitor_id": visitor_id}))
            count = len(existing_embeddings)
            
            if count <= 10:
                print(f"Debug: Existing visitor embeddings count for {visitor_id}: {count}")
                # Cek similarity dengan embedding yang sudah ada
                similarity = self.find_visitor_closest_match(embedding)[1]
                print(f"Debug: Similarity dengan embedding existing: {similarity}")
                
                # Jika similarity > 0.85, embedding terlalu mirip, jangan tambah
                if similarity is None or similarity > 0.85:
                    return False
                
                document = {
                    "visitor_id": ObjectId(visitor_id),
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "created_at": datetime.now().isoformat()
                }
                visitor_vector_collection.insert_one(document)
                return True
            else:
                return False
        else:
            # Fallback ke memory storage
            existing = [d for d in self._memory_storage if d.get("visitor_id") == visitor_id]
            count = len(existing)
            
            if count == 0:
                document = {
                    "visitor_id": visitor_id,
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "created_at": datetime.now()
                }
                self._memory_storage.append(document)
                return True
            else:
                return False
            
    def add_visitor_embedding(self, visitor_id, embedding):
        """
        Tambah embedding baru jika visitor belum memiliki embedding,
        atau jika embedding yang ada sudah berbeda signifikan

        Args:
            visitor_id: ID visitor
            embedding: Face embedding numpy array

        Returns:
            bool: True jika embedding ditambahkan, False jika tidak
        """
        if visitor_vector_collection is not None:
            document = {
                "visitor_id": ObjectId(visitor_id),
                "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "created_at": datetime.now().isoformat()
            }
            visitor_vector_collection.insert_one(document)
            return True
        else:
            # Fallback ke memory storage
            existing = [d for d in self._memory_storage if d.get("visitor_id") == visitor_id]
            count = len(existing)
            
            if count == 0:
                document = {
                    "visitor_id": ObjectId(visitor_id),
                    "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    "created_at": datetime.now().isoformat()
                }
                #sda
                self._memory_storage.append(document)
                return True
            else:
                return False
            
    def add_new_visitor(self, embedding):
        """
        Tambah visitor baru ke database
        
        Args:
            embedding: Face embedding numpy array
            
        Returns:
            visitor_id dari visitor yang baru ditambahkan
        """
        visitor_count = visitor_collection.count_documents({"name": {"$regex": r"^Visitor\d+$"}})
        new_visitor_name = f"Visitor{visitor_count + 1}"
        visitor_doc = {
            "name": new_visitor_name,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "appearance_count": 1
        }
        
        if visitor_collection is not None:
            result = visitor_collection.insert_one(visitor_doc)
            visitor_id = str(result.inserted_id)
        else:
            # Fallback ke memory storage
            visitor_id = str(len(self._memory_storage) + 1)
            visitor_doc["_id"] = visitor_id
            self._memory_storage.append(visitor_doc)
        
        # Simpan embedding
        self.add_visitor_embedding(visitor_id, embedding)
        
        return visitor_id

    
    def close(self):
        """
        Tutup koneksi database
        """
        if self.client is not None:
            self.client.close()
            print("Koneksi MongoDB ditutup.")