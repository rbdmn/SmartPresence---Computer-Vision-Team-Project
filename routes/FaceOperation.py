from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
import logging
from Model.face_recognition_system import FaceRecognitionSystem

router = APIRouter()
system = FaceRecognitionSystem()

class FaceUploadRequest(BaseModel):
    class_id: str
    image_base64: str

@router.post("/face/upload")
async def upload_face_image(payload: FaceUploadRequest):
    try:
        # pakai base64 string
        predicted_user_id = system.recognize_from_base64(payload.image_base64)
        
        return {"status": "success", "results": predicted_user_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.post("/face/uploadmany")
async def upload_face_image_many(payload: FaceUploadRequest):
    try:
        print("Received base64 image for multiple recognition")
        # pakai base64 string
        results = system.recognize_from_base64_many(payload.image_base64, payload.class_id)
        
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/RegisterFaceFromFolder")
async def register_face_from_folder():
    try:
        stats = system.register_faces_from_folder()
        return {
            "status": "success",
            "message": "Faces registered from folder successfully",
            "data": {
                "total_images": stats["total_images"],
                "success": stats["success"],
                "failed": stats["failed"],
                "total_persons": len(stats["persons"]),
                "persons": sorted(stats["persons"])
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}