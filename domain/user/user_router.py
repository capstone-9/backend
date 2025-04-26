from datetime import timedelta, datetime

import os

from fastapi import APIRouter, HTTPException
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from starlette import status

from database import get_db
from domain.user import user_crud, user_schema
from domain.user.user_crud import pwd_context

from starlette.config import Config

config = Config('.env')
ACCESS_TOKEN_EXPIRE_MINUTES = int(config('ACCESS_TOKEN_EXPIRE_MINUTES'))
SECRET_KEY = config('SECRET_KEY')
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/user/login")

router = APIRouter(
    prefix="/api/user",
)


def create_user_folder(user_name: str):
    # 현재 파일의 경로를 기준으로 상대경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리
    storage_dir = os.path.join(base_dir, "../../image")  # 상대경로로 image 폴더 경로 지정
    user_folder = os.path.join(storage_dir, user_name)

    # 유저 이름으로 된 폴더가 없으면 생성
    os.makedirs(user_folder, exist_ok=True)

    return user_folder



@router.post("/create", status_code = status.HTTP_204_NO_CONTENT)
def user_create(_user_create: user_schema.UserCreate, db: Session = Depends(get_db)):
    user1 = user_crud.get_existing_user_name(db, user_create = _user_create)
    user2 = user_crud.get_existing_user_email(db, user_create = _user_create)
    if user2:
        raise HTTPException(status_code=status.HTTP_423_LOCKED,
                            detail="이미 사용중인 이메일입니다.")
    if user1:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="이미 사용중인 아이디입니다.")
    user_crud.create_user(db=db, user_create=_user_create)


@router.post("/login", response_model=user_schema.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
                           db: Session = Depends(get_db)):
    # check user and password
    user = user_crud.get_user(db, form_data.username)
    if not user or not pwd_context.verify(form_data.password, user.password):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # make access token
    data = {
        "sub" : user.username,
        "exp" : datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "access_token" : access_token,
        "token_type" : "bearer",
        "username" : user.username
    }

def get_current_user(token:str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code = status.HTTP_401_UNAUTHORIZED,
        detail = "Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username : str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    else:
        user = user_crud.get_user(db, username=username)
        if user is None:
            raise credentials_exception
        return user
