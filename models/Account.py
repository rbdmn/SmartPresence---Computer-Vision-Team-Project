from pydantic import BaseModel
from typing import List, Optional

class JadwalMengajar(BaseModel):
    nama_mk: str
    waktu_mulai: str  # ISO time string
    waktu_selesai: str  # ISO time string

class Account(BaseModel):
    id: str
    nama: str
    akun_upi: str
    jabatan: str
    jadwal_mengajar: Optional[List[JadwalMengajar]] = []

class LoginRequest(BaseModel):
    akun_upi: str
    password: str
