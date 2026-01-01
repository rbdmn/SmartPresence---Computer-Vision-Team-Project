from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import base64
import logging
from Model.face_recognition_system import FaceRecognitionSystem
from Model.database import FaceDatabase
import time

router = APIRouter()
system = FaceRecognitionSystem()
database = FaceDatabase()

def timerawal():
    return time.perf_counter()

def timerstoptampilkantimer(start):
    elapsed = time.perf_counter() - start
    formatted = f"{elapsed*1000:.2f} ms"
    print("Elapsed:", formatted)
    return formatted

class FaceUploadRequest(BaseModel):
    class_id: str
    image_base64: str

class OpenSetTestRequest(BaseModel):
    registered_folder: Optional[str] = None
    unknown_folder: Optional[str] = None
    thresholds: Optional[List[float]] = None
    threshold: Optional[float] = None

class PrepareDatasetRequest(BaseModel):
    registered_count: Optional[int] = 20

class GroundTruthTestRequest(BaseModel):
    ground_truth_file: str
    thresholds: Optional[List[float]] = None
    threshold: Optional[float] = None
    

@router.get("/faces_embedding")
async def get_faces_embedding():
    try:
        database.get_all_embeddings()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        start = timerawal()
        results = system.recognize_from_base64_many(payload.image_base64, payload.class_id)
        print("Rekognisi Selesai, waktu yang dibutuhkan:")
        timerstoptampilkantimer(start)
        
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

@router.post("/face/prepare_dataset")
async def prepare_openset_dataset(payload: PrepareDatasetRequest):
    """
    Prepare open-set dataset by splitting ChokePoint into registered and unknown
    """
    try:
        print(f"Preparing open-set dataset with {payload.registered_count} registered users")
        start = timerawal()
        
        result = system.prepare_openset_dataset(
            registered_count=payload.registered_count
        )
        
        print("Dataset preparation completed, waktu yang dibutuhkan:")
        elapsed = timerstoptampilkantimer(start)
        
        return {
            "status": "success",
            "message": "Open-set dataset prepared successfully",
            "data": result,
            "elapsed_time": elapsed
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/face/test_openset")
async def test_open_set_accuracy(payload: OpenSetTestRequest):
    """
    Test open-set recognition accuracy with TAR, FAR, FRR, TRR metrics
    """
    try:
        print("Starting open-set evaluation...")
        start = timerawal()
        
        results = system.test_open_set_accuracy(
            registered_folder=payload.registered_folder,
            unknown_folder=payload.unknown_folder,
            thresholds=payload.thresholds,
            threshold=payload.threshold
        )
        
        print("Evaluation completed, waktu yang dibutuhkan:")
        elapsed = timerstoptampilkantimer(start)
        
        return {
            "status": "success",
            "message": "Open-set evaluation completed",
            "data": results,
            "elapsed_time": elapsed
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/face/test_openset_groundtruth")
async def test_openset_with_groundtruth(payload: GroundTruthTestRequest):
    """
    Test open-set accuracy dengan ground truth manual
    Untuk testing dengan real CCTV frames + unknown people dari LFW
    """
    try:
        print(f"Starting ground truth-based evaluation...")
        print(f"Ground truth file: {payload.ground_truth_file}")
        start = timerawal()
        
        results = system.test_open_set_with_groundtruth(
            ground_truth_file=payload.ground_truth_file,
            thresholds=payload.thresholds,
            threshold=payload.threshold
        )
        
        print("Evaluation completed, waktu yang dibutuhkan:")
        elapsed = timerstoptampilkantimer(start)
        
        return {
            "status": "success",
            "message": "Ground truth-based evaluation completed",
            "data": results,
            "elapsed_time": elapsed
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))