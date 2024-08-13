import uuid
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, List

class DeployResponse(BaseModel):
    deployed_id: str
    message: str

class HTTPExceptionResponse(BaseModel):
    status_code: int
    message: str
    data: Optional[str] = None

class SingleClassifierClass(BaseModel):
    name: str
    examples_to_include: List[str]
    examples_to_exclude: List[str] = []
    
    class Config:
        orm_mode = True
