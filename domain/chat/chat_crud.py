from sqlalchemy.orm import Session
from models import Diary, User
from datetime import datetime
from domain.chat import chat_schema


def create_diary(
        db: Session,
        username: str,
        summary: str,
        emotion: str,
        image_url: str,
        date: datetime  # YYYY-MM-DD 형식 문자열
):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None

    diary = Diary(
        user_id=user.id,
        date=date,
        summary=summary,
        emotion=emotion,
        image_url=image_url
    )
    db.add(diary)
    db.commit()
    db.refresh(diary)
    return diary


def get_diaries_by_user(db: Session, username: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return []

    return db.query(Diary).filter(Diary.user_id == user.id).order_by(Diary.date.desc()).all()
