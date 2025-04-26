from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from starlette.config import Config
from starlette import status
from database import get_db
from sqlalchemy.orm import Session
from domain.chat import chat_schema, chat_crud
from domain.user.user_router import get_current_user

from datetime import datetime
from fastapi.responses import FileResponse
import os
import json
import base64

# LangChain + Ollama
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema.runnable import RunnableSequence


from domain.AImodels.bert import analyze_emotion
from domain.AImodels.stablediffusion import generate_image



router = APIRouter(
    prefix="/api/chat",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/user/login")

def clean_json_block(text):
    # 앞뒤로 ```json 이나 ``` 같은 코드 블럭 태그 있으면 삭제
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()



llm =OllamaLLM(model="exaone3.5")

# 프롬프트: 자연스럽고 친근한 말투로 응답
chat_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
    너는 사용자의 가장 친한 친구야. 따뜻하고 친근한 성격을 가지고 있고, 마치 실제로 만나서 이야기하는 것처럼 자연스럽게 반응해.
    사용자의 감정(기쁨, 슬픔, 분노, 고민 등)을 읽고, 공감하거나 응원해주거나 다정하게 다독여줘.
    그리고 짧게라도 질문을 덧붙여서 사용자가 자연스럽게 더 이야기하고 싶게 만들어줘.

    - 너무 딱딱하거나 사무적인 말투는 절대 쓰지 마.
    - 적당히 가볍고 따뜻한 친구 말투를 써.
    - 문장 길이는 1~3문장 정도로 자연스럽게.
    - 필요하면 이모지도 살짝 써도 좋아.

    사용자의 말:
    {input}

    이 말에 대해 친구처럼 자연스럽게 공감하고, 짧은 질문도 덧붙여서 답변해줘.
    """
)
chat_chain = RunnableSequence(chat_prompt | llm)

summary_prompt = PromptTemplate(
    input_variables=["messages"],
    template="""
    다음은 사용자의 대화 내용이야. 이걸 바탕으로 간단한 일기처럼 정리해줘.

    - 사용자가 실제로 한 말만 바탕으로 써줘. 상상하거나 추측하지 마.
    - 핵심적인 활동, 장소, 기분 정도만 담백하게 정리해줘.
    - 1인칭 시점으로 써줘.
    - 너무 시적이거나 문학적인 표현은 피하고, 자연스럽고 따뜻하게 하루를 회상하는 느낌이면 좋아.
    - "오늘은"으로 시작해줘.

    사용자 메시지:
    {messages}
    """
)
summary_chain = RunnableSequence(summary_prompt | llm)

keyword_prompt = PromptTemplate(
    input_variables=["messages"],
    template="""
    다음 대화에서 핵심적인 단어들만 뽑아줘. 
    - 구체적인 장소, 날씨만 뽑아줘.
    - 장소는 대화를 분석해보고 너가 생각하기에 가장 임팩트가 큰 장소 하나만 뽑아줘.
    - 날씨도 대화를 분석해보고 너가 생각하기에 가장 임팩트가 큰 날씨 하나만 뽑아줘.
    - 반드시 **단어만** 뽑고, **문장은 절대 포함하지 마**.
    - 사용자가 실제로 언급한 단어만 뽑아줘.

    결과는 **반드시 JSON 배열**로 줘.
    절대로 다른 말(예: 설명, 안내문) 붙이지 말고, JSON 배열만 출력해.

    예시: ["맑음", "학교"]

    대화 내용:
    {messages}
    """
)
keyword_chain = RunnableSequence(keyword_prompt | llm)

eng_prompt = PromptTemplate(
    input_variables=["messages"],
    template="""
    다음 한국어 단어들을 영어로 번역해줘. 
    - 각 단어를 간단하고 자연스럽게 번역해.
    - 문장이나 해석은 절대 포함하지 마.
    - 반드시 단어만 번역하고, 결과는 JSON 배열로만 줘.

    예시: ["맑음", "학교"] → ["sunny", "school"]

    번역할 단어:
    {messages}
    """
)
eng_chain = RunnableSequence(eng_prompt | llm)

# POST 라우터
@router.post("/talk", response_model=chat_schema.ChatResponse)
def chat(request: chat_schema.ChatRequest):
    try:
        result = chat_chain.invoke({"input": request.message})
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챗봇 응답 생성 실패: {e}")

@router.post("/save", response_model=chat_schema.ChatSaveResponse)
def save(request: chat_schema.ChatSave,
         db: Session = Depends(get_db),
         current_user=Depends(get_current_user)
         ):
    full_conversation = "\n".join([f"{m['speaker']}: {m['message']}" for m in request.conversation])

    # 요약 생성
    user_messages_only = "\n".join([m["message"] for m in request.conversation if m["speaker"] == "user"])
    summary = summary_chain.invoke({"messages": user_messages_only})

    # 키워드 생성
    keyword = keyword_chain.invoke({"messages": full_conversation})
    keyword = clean_json_block(keyword)
    try:
        keyword = json.loads(keyword)
    except json.JSONDecodeError:
        print("⚠️ 키워드 파싱 실패! 원본 출력:", keyword)

    # 영어 생성
    english = eng_chain.invoke({"messages": str(keyword)})
    english = clean_json_block(english)
    try:
        english = json.loads(english)
    except json.JSONDecodeError:
        print("⚠️ 영어 파싱 실패! 원본 출력:", english)

    emotion = analyze_emotion(summary)

    prompt = "The day was " + english[0] + " and I went to the " + english[1]
    today_str = datetime.now().strftime("%Y-%m-%d")
    date_obj = datetime.strptime(today_str, "%Y-%m-%d")
    filename = f"{current_user.username}_{today_str}.png"
    image_path = generate_image([prompt], output_path="C:/projects/haru/image/"+filename)


    saved_diary = chat_crud.create_diary(
        db=db,
        username=current_user.username,
        summary=summary,
        emotion=emotion,
        image_url=image_path,
        date=date_obj
    )

    return {
        "summary": summary,
        "emotion": emotion,
        "imageurl": image_path,
        "date" : date_obj
    }


@router.post("/get-image")
def get_image(request: chat_schema.ChatImage, current_user=Depends(get_current_user)):
    imageurl = request.imageurl

    if not os.path.isfile(imageurl):
        raise HTTPException(status_code=404, detail="이미지 파일을 찾을 수 없습니다.")

    return FileResponse(
        path=imageurl,
        media_type="image/png",
        filename=os.path.basename(imageurl)
    )


@router.get("/diaries", response_model=list[chat_schema.DiaryResponse])
def get_diaries(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    diaries = chat_crud.get_diaries_by_user(db, current_user.username)
    return diaries
