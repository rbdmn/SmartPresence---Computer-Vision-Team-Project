
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from starlette.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from config.configrations import attendance_collection, users_collection, account_collection, class_collection
from models.Account import Account
from models.Users import UserModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt

SECRET_KEY = "your_secret_key"  # Use the same as in Account router
ALGORITHM = "HS256"

router = APIRouter()

# JWT dependency (reuse from Account router if possible)
bearer_scheme = HTTPBearer()
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        account_id: str = payload.get("account_id")
        jabatan: str = payload.get("jabatan")
        if account_id is None or jabatan is None:
            raise credentials_exception
        # Get full account info
        acc = account_collection.find_one({"_id": ObjectId(account_id)})
        if not acc:
            raise credentials_exception
        acc["id"] = str(acc["_id"])
        return acc
    except Exception:
        raise credentials_exception

# --- Attendance Report Endpoint ---

# --- Scenario A: By Schedule (auto) ---

@router.get("/attendance/report/by-schedule")
async def attendance_report_by_schedule(
    course_name: str = Query(..., description="Nama mata kuliah (course name)"),
    meeting_date: str = Query(..., description="Tanggal pertemuan (YYYY-MM-DD)"),
    current_user: dict = Depends(get_current_user)
):
    jadwal = None
    for j in current_user.get("jadwal_mengajar", []):
        if j.get("nama_mk") == course_name:
            jadwal = j
            break
    if not jadwal:
        raise HTTPException(status_code=404, detail="Course not found in lecturer's schedule")
    waktu_mulai = jadwal["waktu_mulai"]
    waktu_selesai = jadwal["waktu_selesai"]
    class_id = jadwal.get("class_id")
    if not class_id:
        raise HTTPException(status_code=400, detail="class_id not found in jadwal_mengajar")
    # If class_id is already ObjectId, use as is; if string, convert
    if isinstance(class_id, ObjectId):
        class_obj_id = class_id
    else:
        try:
            class_obj_id = ObjectId(class_id)
        except Exception:
            raise HTTPException(status_code=400, detail="class_id in jadwal_mengajar is not a valid ObjectId")
    if not class_collection.find_one({"_id": class_obj_id}):
        raise HTTPException(status_code=404, detail="Class not found")
    try:
        start_time = datetime.strptime(f"{meeting_date} {waktu_mulai}", "%Y-%m-%d %H:%M")
        end_time = datetime.strptime(f"{meeting_date} {waktu_selesai}", "%Y-%m-%d %H:%M")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date/time: {e}")
    print(f"[DEBUG] BY SCHEDULE: course_name={course_name}, meeting_date={meeting_date}, start_time={start_time}, end_time={end_time}, class_id={class_id}")
    pipeline = [
        {"$addFields": {
            "timestamp_date": {"$toDate": "$timestamp"}
        }},
        {"$match": {
            "timestamp_date": {"$gte": start_time, "$lte": end_time},
            "$or": [
                {"class_id": str(class_obj_id)},
                {"class_id": class_obj_id}
            ]
        }},
        {"$lookup": {
            "from": "Users",
            "localField": "user_id",
            "foreignField": "_id",
            "as": "student_info"
        }},
        {"$lookup": {
            "from": "Visitors",
            "localField": "visitor_id",
            "foreignField": "_id",
            "as": "visitor_info"
        }},
        {"$project": {
            "timestamp": 1,
            "full_name": {
                "$cond": [
                    {"$gt": [{"$size": "$student_info"}, 0]},
                    {"$arrayElemAt": ["$student_info.name", 0]},
                    {"$arrayElemAt": ["$visitor_info.name", 0]}
                ]
            },
            "category": {
                "$cond": [
                    {"$gt": [{"$size": "$student_info"}, 0]},
                    "Mahasiswa",
                    "Visitor"
                ]
            }
        }}
    ]
    results = list(attendance_collection.aggregate(pipeline))
    print(f"[DEBUG] Matched attendance records: {len(results)}")
    if results:
        print(f"[DEBUG] First result: {results[0]}")
    print(f"[DEBUG] Attendees: {[r.get('full_name', 'Unknown') for r in results]}")
    for r in results:
        # Convert ObjectId fields to string
        for k, v in r.items():
            if isinstance(v, ObjectId):
                r[k] = str(v)
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].isoformat()
        # Ensure full_name and category exist
        if 'full_name' not in r:
            r['full_name'] = 'Unknown'
        if 'category' not in r:
            r['category'] = 'Unknown'
    return {
        "scenario": "by-schedule",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_attendance": len(results),
        "attendees": results
    }

# --- Scenario B: Manual (custom time range) ---

@router.get("/attendance/report/by-manual")
async def attendance_report_by_manual(
    class_id: str = Query(..., description="ID kelas (class_id)"),
    specific_date: str = Query(..., description="Tanggal (YYYY-MM-DD)"),
    start_time_str: str = Query(..., description="Jam mulai (HH:mm)"),
    end_time_str: str = Query(..., description="Jam selesai (HH:mm)"),
    # current_user: dict = Depends(get_current_user)
):
    try:
        start_time = datetime.strptime(f"{specific_date} {start_time_str}", "%Y-%m-%d %H:%M")
        end_time = datetime.strptime(f"{specific_date} {end_time_str}", "%Y-%m-%d %H:%M")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date/time: {e}")
    # Validate class_id is a valid ObjectId format and exists in Class collection
    try:
        class_obj_id = ObjectId(class_id)
    except Exception:
        raise HTTPException(status_code=400, detail="class_id is not a valid ObjectId")
    if not class_collection.find_one({"_id": class_obj_id}):
        raise HTTPException(status_code=404, detail="Class not found")
    print(f"[DEBUG] BY MANUAL: class_id={class_id}, specific_date={specific_date}, start_time={start_time}, end_time={end_time}")
    
    # Check if class_id in attendance is stored as string or ObjectId
    # Try to match both string and ObjectId format
    pipeline = [
        {"$addFields": {
            "timestamp_date": {"$toDate": "$timestamp"}
        }},
        {"$match": {
            "timestamp_date": {"$gte": start_time, "$lte": end_time},
            "$or": [
                {"class_id": class_id},  # Match as string
                {"class_id": class_obj_id}  # Match as ObjectId
            ]
        }},
        {"$lookup": {
            "from": "Users",
            "localField": "user_id",
            "foreignField": "_id",
            "as": "student_info"
        }},
        {"$lookup": {
            "from": "Visitors",
            "localField": "visitor_id",
            "foreignField": "_id",
            "as": "visitor_info"
        }},
        {"$project": {
            "timestamp": 1,
            "full_name": {
                "$cond": [
                    {"$gt": [{"$size": "$student_info"}, 0]},
                    {"$arrayElemAt": ["$student_info.name", 0]},
                    {"$arrayElemAt": ["$visitor_info.name", 0]}
                ]
            },
            "category": {
                "$cond": [
                    {"$gt": [{"$size": "$student_info"}, 0]},
                    "Mahasiswa",
                    "Visitor"
                ]
            }
        }}
    ]
    results = list(attendance_collection.aggregate(pipeline))
    print(f"[DEBUG] Matched attendance records: {len(results)}")
    if results:
        print(f"[DEBUG] First result: {results[0]}")
    print(f"[DEBUG] Attendees: {[r.get('full_name', 'Unknown') for r in results]}")
    for r in results:
        # Convert ObjectId fields to string
        for k, v in r.items():
            if isinstance(v, ObjectId):
                r[k] = str(v)
        if isinstance(r.get("timestamp"), datetime):
            r["timestamp"] = r["timestamp"].isoformat()
        # Ensure full_name and category exist
        if 'full_name' not in r:
            r['full_name'] = 'Unknown'
        if 'category' not in r:
            r['category'] = 'Unknown'
    return {
        "scenario": "by-manual",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_attendance": len(results),
        "attendees": results
    }

# ===============================================
# IGNORE BELOW: Attendance Logging Endpoint
# ===============================================

class AttendanceLogRequest(BaseModel):
    user_id: str
    status: str
    class_name: str  # Tambahkan field class_name

@router.post("/log")
async def log_attendance(payload: AttendanceLogRequest):
    try:
        user_id_obj = ObjectId(payload.user_id)
    except InvalidId:
        return {"status": "failed", "message": "Invalid user_id format"}

    user = users_collection.find_one({"_id": user_id_obj})
    if not user:
        return {"status": "failed", "message": "User not found"}

    now = datetime.now(timezone.utc)
    class_name = payload.class_name  # Dapatkan class_name dari payload
    attendance_collection.insert_one({
        "user_id": user_id_obj,
        "timestamp": now,
        "status": payload.status,
        "class_name": class_name  # Simpan class_name ke dalam attendance
    })

    return {
        "status": "success",
        "timestamp": now.isoformat()
    }