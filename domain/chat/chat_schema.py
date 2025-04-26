from pydantic import BaseModel
from datetime import datetime

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class ChatSave(BaseModel):
    conversation : list[dict]

class ChatSaveResponse(BaseModel):
    date : datetime
    summary : str
    emotion : str
    imageurl : str

class ChatImage(BaseModel):
    imageurl : str

class DiaryResponse(BaseModel):
    id: int
    summary: str
    emotion: str
    image_url: str
    date: datetime