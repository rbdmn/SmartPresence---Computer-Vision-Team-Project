from fastapi import APIRouter, HTTPException
from starlette.responses import JSONResponse
from models.Users import UserModel, VisitorModel
from typing import List
from config.configrations import users_collection, visitor_collection
from datetime import datetime
from bson import ObjectId

router = APIRouter()

@router.get("/", response_model=List[UserModel])
async def get_users():
    users = list(users_collection.find().limit(100))  # Limit to 100 users
    for user in users:
        user["_id"] = str(user["_id"])  # Convert ObjectId to string for JSON serialization
        if "created_at" in user and isinstance(user["created_at"], datetime):
            user["created_at"] = user["created_at"].isoformat()  # Convert datetime to ISO format
    return JSONResponse(content=users)


@router.get("/{user_id}", response_model=UserModel)
async def get_user_by_id(user_id: str):
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user["_id"] = str(user["_id"])
    if "created_at" in user and isinstance(user["created_at"], datetime):
        user["created_at"] = user["created_at"].isoformat()
    
    return JSONResponse(content=user)

@router.get("/visitor/{visitor_id}", response_model=VisitorModel)
async def get_visitor_by_id(visitor_id: str):
    try:
        print(f"[DEBUG] Fetching visitor with ID: {visitor_id}")
        visitor = visitor_collection.find_one({"_id": ObjectId(visitor_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid visitor ID format")
    
    if not visitor:
        raise HTTPException(status_code=404, detail="visitor not found")
    
    visitor["_id"] = str(visitor["_id"])
    if "first_seen" in visitor and isinstance(visitor["first_seen"], datetime):
        visitor["first_seen"] = visitor["first_seen"].isoformat()
    
    return JSONResponse(content=visitor)