from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class ChatSave(BaseModel):
    conversation : list[dict]

class ChatSaveResponse(BaseModel):
    date : datetime
    summary : str
    emotion : list[float]
    imageurl : str

class ChatImage(BaseModel):
    imageurl : str

class DiaryResponse(BaseModel):
    id: int
    summary: str
    emotion: list[float]
    image_url: str
    date: datetime
    conversation: Optional[list[dict]] = []