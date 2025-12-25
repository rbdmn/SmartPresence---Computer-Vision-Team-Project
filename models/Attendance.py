from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class AttendanceModel(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    user_id: str
    timestamp: datetime
    status: str

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "user_id": "60c72b2f9b1e8b3a2c8f9e7d",
                "timestamp": "2024-01-01T08:00:00",
                "status": "present"
            }
        }
