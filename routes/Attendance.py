from fastapi import APIRouter, HTTPException
from starlette.responses import JSONResponse
from models.Attendance import AttendanceModel
from typing import List
from config.configrations import attendance_collection, users_collection
from datetime import datetime, timezone
from pydantic import BaseModel
from bson import ObjectId
from bson.errors import InvalidId

router = APIRouter()

@router.get("/", response_model=List[AttendanceModel])
async def get_attendance():
    records = list(attendance_collection.find().limit(100))
    for record in records:
        record["_id"] = str(record["_id"])
        if "user_id" in record:
            record["user_id"] = str(record["user_id"])
        if "timestamp" in record and isinstance(record["timestamp"], datetime):
            record["timestamp"] = record["timestamp"].isoformat()
    return JSONResponse(content=records)

class AttendanceLogRequest(BaseModel):
    user_id: str
    status: str

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
    attendance_collection.insert_one({
        "user_id": user_id_obj,
        "timestamp": now,
        "status": payload.status
    })

    return {
        "status": "success",
        "timestamp": now.isoformat()
    }