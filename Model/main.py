"""
Main script untuk menjalankan Face Recognition System
"""
import argparse
import cv2
from pathlib import Path
from .face_recognition_system import FaceRecognitionSystem
from .config import DATABASE_IMAGES_DIR, RECOGNITION_THRESHOLD, TESTING_IMAGES_DIR


def main():
    parser = argparse.ArgumentParser(description='Face Recognition Attendance System')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['register', 'test', 'recognize', 'stats', 'full', 'demo'],
                       help='Mode operasi')
    parser.add_argument('--image', type=str, help='Path gambar untuk recognition')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold untuk cosine similarity')
    parser.add_argument('--db-folder', type=str, default=str(DATABASE_IMAGES_DIR),
                       help='Folder untuk database images')
    parser.add_argument('--test-folder', type=str, default=str(TESTING_IMAGES_DIR),
                       help='Folder untuk testing images')
    
    args = parser.parse_args()
    
    # Inisialisasi sistem
    system = FaceRecognitionSystem()
    
    try:
        if args.mode == 'register':
            # Hanya registrasi wajah
            system.register_faces_from_folder(args.db_folder)
            
        elif args.mode == 'test':
            # Test akurasi
            system.test_accuracy(args.test_folder, RECOGNITION_THRESHOLD)
            
        elif args.mode == 'recognize':
            # Recognize single image
            if not args.image:
                print("Error: --image diperlukan untuk mode recognize")
                return
            
            results = system.recognize_from_file(args.image, RECOGNITION_THRESHOLD)
            
            if not results:
                print("Tidak ada wajah terdeteksi")
            else:
                for i, result in enumerate(results):
                    print(f"Wajah {i+1}: {result['name']} (distance: {result['distance']:.3f})")
                
                # Show image with detection
                image = cv2.imread(args.image)
                for result in results:
                    image = system.detector.draw_detection(
                        image, 
                        result['face'], 
                        result['name'], 
                        result['distance']
                    )
                
                cv2.imshow('Recognition Result', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        elif args.mode == 'stats':
            # Tampilkan statistik database
            stats = system.get_database_stats()
            print(f"\n--- Database Statistics ---")
            print(f"Total embeddings: {stats['total_embeddings']}")
            print(f"Total persons: {stats['total_persons']}")
            print(f"\nDetail per person:")
            for person, count in stats['persons'].items():
                print(f"  - {person}: {count} embeddings")
                
        elif args.mode == 'full':
            # Full pipeline: register + test
            print("\n" + "="*60)
            print("FASE 1: REGISTRASI WAJAH")
            print("="*60)
            system.register_faces_from_folder(args.db_folder)
            
            print("\n" + "="*60)
            print("FASE 2: STATISTIK DATABASE")
            print("="*60)
            stats = system.get_database_stats()
            print(f"Total embeddings: {stats['total_embeddings']}")
            print(f"Total persons: {stats['total_persons']}")
            
            # Check if testing folder has images
            test_folder = Path(args.test_folder)
            if test_folder.exists() and any(test_folder.iterdir()):
                print("\n" + "="*60)
                print("FASE 3: TESTING AKURASI")
                print("="*60)
                system.test_accuracy(args.test_folder, RECOGNITION_THRESHOLD)
            else:
                print(f"\nFolder testing kosong: {args.test_folder}")
                print("Skip testing akurasi")
                
        elif args.mode == 'demo':
            # Demo interaktif
            demo_interactive(system, RECOGNITION_THRESHOLD)
    
    finally:
        system.close()


def demo_interactive(system: FaceRecognitionSystem, threshold: float):
    """
    Demo interaktif dengan menu
    """
    while True:
        print("\n" + "="*40)
        print("FACE RECOGNITION DEMO")
        print("="*40)
        print("1.Register wajah dari folder")
        print("2.Test akurasi")
        print("3.Recognize gambar")
        print("4.Lihat statistik database")
        print("5.Hapus database")
        print("6.Keluar")
        print("="*40)
        
        choice = input("Pilih menu: ").strip()
        
        if choice == '1':
            folder = input(f"Folder (default: {DATABASE_IMAGES_DIR}): ").strip()
            if not folder:
                folder = str(DATABASE_IMAGES_DIR)
            system.register_faces_from_folder(folder)
            
        elif choice == '2':
            folder = input(f"Folder testing (default: {TESTING_IMAGES_DIR}): ").strip()
            if not folder:
                folder = str(TESTING_IMAGES_DIR)
            system.test_accuracy(folder, threshold)
            
        elif choice == '3':
            image_path = input("Path gambar: ").strip()
            if image_path:
                results = system.recognize_from_file(image_path, threshold)
                if results:
                    for r in results:
                        print(f"  -> {r['name']} (distance: {r['distance']:.3f})")
                else:
                    print("Tidak ada wajah terdeteksi")
                    
        elif choice == '4':
            stats = system.get_database_stats()
            print(f"Total embeddings: {stats['total_embeddings']}")
            print(f"Total persons: {stats['total_persons']}")
            for person, count in stats['persons'].items():
                print(f"  - {person}: {count}")
                
        elif choice == '5':
            confirm = input("Yakin hapus semua data? (y/n): ").strip().lower()
            if confirm == 'y':
                deleted = system.database.clear_database()
                print(f"Dihapus {deleted} entri")
                
        elif choice == '6':
            print("Bye!")
            break


if __name__ == "__main__":
    main()