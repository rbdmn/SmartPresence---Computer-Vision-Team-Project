# models/user.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class UserModel(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    name: str
    created_at: datetime

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "name": "John Doe",
                "created_at": "2024-01-01T12:00:00"
            }
        }
        
class VisitorModel(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    name: str
    first_seen: datetime

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "name": "John Doe",
                "first_seen": "2024-01-01T12:00:00"
            }
        }
# Note: The 'id' field is optional and uses an alias to map to MongoDB's '_id' field.