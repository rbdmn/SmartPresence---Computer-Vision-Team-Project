from fastapi import APIRouter
from starlette.responses import JSONResponse
from models.Vector import VectorModel
from typing import List
from config.configrations import vector_collection

router = APIRouter()

@router.get("/", response_model=List[VectorModel])
async def get_vectors():
    vectors = list(vector_collection.find().limit(100))
    for vector in vectors:
        vector["_id"] = str(vector["_id"])
        if "user_id" in vector:
            vector["user_id"] = str(vector["user_id"])
    return JSONResponse(content=vectors)
