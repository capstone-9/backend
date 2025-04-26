from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from datetime import datetime

from database import Base

class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)

    diaries = relationship("Diary", back_populates="user")

class Diary(Base):
    __tablename__ = "diary"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    summary = Column(Text)
    emotion = Column(String)  # 예: "기쁨", "슬픔"
    image_url = Column(String)

    user = relationship("User", back_populates="diaries")