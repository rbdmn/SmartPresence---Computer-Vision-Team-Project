"""
Simple test script untuk memverifikasi sistem berjalan dengan benar
"""
from .face_recognition_system import FaceRecognitionSystem
from .config import DATABASE_IMAGES_DIR, TESTING_IMAGES_DIR


def simple_test():
    """
    Test sederhana untuk memastikan semua komponen bekerja
    """
    print("="*60)
    print("SIMPLE TEST - Face Recognition System")
    print("="*60)
    
    # 1. Inisialisasi sistem
    print("\n[1] Inisialisasi sistem...")
    system = FaceRecognitionSystem()
    
    # 2.  Registrasi wajah
    print("\n[2] Registrasi wajah dari database_images/...")
    try:
        stats = system.register_faces_from_folder()
        print(f"    Berhasil mendaftarkan {stats['success']} wajah dari {len(stats['persons'])} orang")
    except ValueError as e:
        print(f"    Warning: {e}")
        print("    Pastikan folder database_images/ berisi gambar wajah!")
        return
    
    # 3.  Cek database
    print("\n[3] Cek statistik database...")
    db_stats = system. get_database_stats()
    print(f"    Total embeddings: {db_stats['total_embeddings']}")
    print(f"    Total persons: {db_stats['total_persons']}")
    
    # 4. Test recognition
    print("\n[4] Test recognition...")
    try:
        test_results = system.test_accuracy()
        print(f"    Akurasi: {test_results['accuracy']:.2f}%")
    except ValueError as e:
        print(f"    Warning: {e}")
        print("    Pastikan folder testing_images/ berisi gambar untuk testing!")
    
    # 5. Cleanup
    system.close()
    
    print("\n" + "="*60)
    print("TEST SELESAI!")
    print("="*60)


if __name__ == "__main__":
    simple_test()