import os
import re

def natural_sort_key(s):
    """
    Fungsi bantuan untuk mengurutkan file secara alami.
    Agar urutannya: 1.jpg, 2.jpg, ..., 10.jpg (Bukan 1.jpg, 10.jpg, 2.jpg)
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def rename_images():
    # 1. Input Nama
    folder_path = input("Path folder (Enter untuk folder ini): ").strip()
    if not folder_path:
        folder_path = "."
        
    base_name = input("Masukkan Nama (contoh: Danes): ").strip()
    
    if not os.path.exists(folder_path):
        print("Folder tidak ditemukan!")
        return

    # Ambil semua file gambar
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 2. Urutkan file secara 'Natural' agar urutan 1, 2, 10 benar
    files.sort(key=natural_sort_key)
    
    print(f"\nDitemukan {len(files)} gambar. Memulai rename...\n")

    # 3. Loop dengan enumerate (mulai hitungan dari 1)
    for index, filename in enumerate(files, start=1):
        # Ambil ekstensi asli (.jpg/.png)
        ext = os.path.splitext(filename)[1]
        
        # Format Baru: [NAMA]_[ITERASI][EKSTENSI]
        # Contoh: Danes_1.jpg
        new_name = f"{base_name}_{index}{ext}"
        
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # Cek agar tidak menimpa jika nama sudah sama
        if old_file != new_file:
            try:
                os.rename(old_file, new_file)
                print(f"[OK] {filename} -> {new_name}")
            except OSError as e:
                print(f"[ERR] Gagal rename {filename}: {e}")
        else:
            print(f"[SKIP] {filename} sudah memiliki nama yang sesuai.")

    print("\nSelesai!")

if __name__ == "__main__":
    rename_images()