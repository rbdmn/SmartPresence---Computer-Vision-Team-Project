from pydantic import BaseModel, Field
from typing import Optional, List

class VectorModel(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    user_id: str
    array_vector: list
    similarity: float

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {
                "user_id": "60c72b2f9b1e8b3a2c8f9e7d",
                "array_vector": [0.1, 0.2, 0.3, 0.4],
                "similarity": 0.95
            }
        }
