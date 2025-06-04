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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_json_block(text):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1]
    elif "```" in text:
        text = text.split("```")[1]
    elif "\"""" in text:
        text = text.split("\"""")[1]
    elif "**" in text:
        text = text.split("**")[1]
    return text.strip().strip("`")



llm = OllamaLLM(model="exaone3.5:7.8b")

# 프롬프트: 자연스럽고 친근한 말투로 응답
chat_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
    너는 사용자의 가장 친한 친구야. 평소처럼 편하게 얘기 중이야. 지금 너희 둘은 과거 이야기를 나누고 있어.
    사용자가 지금까지 너와 어떤 이야기를 나눴는지 아래에 기록되어 있어. 이걸 바탕으로 공감하고, 사용자의 감정을 이어받아 자연스럽게 질문을 던져줘.

    반드시 지켜야 할 것:
    - 친구끼리 대화하듯 반말로 따뜻하게 말해줘
    - 문장은 20자 이내로 짧고 자연스럽게
    - 감정을 먼저 이어받고, 가볍게 물어보듯 말해
    - 너무 가볍거나 장난스럽지 않게
    - 너무 딱딱하거나 설명하려고 하지 마
    - “헐”, “그랬구나”, “진짜 속상했겠다” 같은 **입말체**를 적절히 써도 좋아
    - 사용자의 감정과 톤에 꼭 맞춰야 해
    - 줄바꿈 없이 한 문장으로 대답
    - 다음 대화를 자연스럽게 유도할 것
    - 특수문자, 이모지, 이모티콘은 절대 사용하지 마.

    지금까지의 대화:
    {history}

    사용자의 말:
    {input}
    """
)
chat_chain = RunnableSequence(chat_prompt | llm)

summary_prompt = PromptTemplate(
    input_variables=["messages"],
    template="""
    다음은 사용자의 대화 내용이야. 이걸 바탕으로 간단한 일기처럼 정리해줘.

    주의:
    - 반드시 사용자가 실제로 한 말만 바탕으로 써줘. 상상하거나 추측하지 마.
    - 사용자가 하지 않은 말은 요약에 사용해서는 안돼.
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
    다음 대화에서 핵심 키워드 또는 주요 활동 내용을 뽑아줘.

    결과는 다음과 같은 JSON 객체 형식으로 출력해줘:
    {{
      "tags": ["맑음", "학교"],
      "activities": ["수업을 하고 옴", "도서관에서 공부함", "집에서 밥 먹음"],
      "summary": ["오늘 맑은 날씨 속에서 학교에서 수업을 듣고, 도서관에서 공부하고 집에서 밥을 먹었다."]
    }}

    주의:
    - 반드시 JSON 객체 하나로 출력할 것.
    - 문장 대신 JSON 배열로 구성할 것.
    - 설명, 안내문은 포함하지 말 것.
    - 문장을 생성할 때, 사용자가 하지 않은 쓸데 없는 말은 추가하지 말 것.

    대화 내용:
    {messages}
    """
)
keyword_chain = RunnableSequence(keyword_prompt | llm)

eng_prompt = PromptTemplate(
    input_variables=["messages"],
    template="""
    다음 한국어 단어 또는 짧은 구문들을 자연스러운 영어로 번역해줘.
    앞서 생성한 keyword_prompt의 tags, activities, summary만 그대로 영어로 번역해줘.


    결과는 다음과 같은 JSON 객체 형식으로 출력해줘:
    {{
      "eng_tags": ["shine", "school"],
      "eng_activities": ["I had a class", "studying in the library", "Eating at home"],
      "eng_summary": ["Today, I took classes at school in clear weather, studied at the library, and ate at home."]
    }}

    주의:
    - 반드시 JSON 객체 하나로 출력할 것.
    - 문장 대신 JSON 배열로 구성할 것.
    - 설명, 안내문은 포함하지 말 것.

    번역할 항목:
    {messages}
    """
)
eng_chain = RunnableSequence(eng_prompt | llm)

sd_prompt = PromptTemplate(
    input_variables=["structured_summary"],
    template="""
    다음은 사용자의 대화에서 추출한 핵심 정보야. 이걸 바탕으로 **Stable Diffusion**에 쓸 수 있는 영어 프롬프트를 **짧고 간결하게 한 문장**으로 만들어줘.

    조건:
    - 반드시 영어로 작성해.
    - 한 문장은 20단어 이내로 작성해.
    - 묘사는 간결하고 명확해야 해. 수식어를 너무 많이 붙이지 마.
    - 감정(emotion)은 색감이나 분위기로 반영해.
    - imagery는 시각적으로 표현해.
    - 1인칭 말투 쓰지 마.
    - 아래 JSON의 모든 요소(subject, location, emotion, event, imagery)는 꼭 포함해.
    - 추가 설명이나 안내문은 쓰지 마.

    JSON:
    {structured_summary}
    """
)
sd_chain = RunnableSequence(sd_prompt | llm)

# POST 라우터
@router.post("/talk", response_model=chat_schema.ChatResponse)
def chat(request: chat_schema.ChatRequest):
    try:
        history_text = "\n".join([f"{m['speaker']}: {m['message']}" for m in request.conversation])
        result = chat_chain.invoke({"history": history_text, "input": request.message})
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
    keyword_raw = keyword_chain.invoke({"messages": full_conversation})
    keyword_clean = clean_json_block(keyword_raw)
    try:
        keyword_json = json.loads(keyword_clean)
        tags = keyword_json.get("tags", [])
        activities = keyword_json.get("activities", [])
        summary_sentences = keyword_json.get("summary", [])
    except json.JSONDecodeError:
        print("⚠️ 키워드 파싱 실패! 원본 출력:", keyword_clean)
        tags, activities, summary_sentences = [], [], []

    # 영어 번역
    translation_targets = {
        "tags": tags,
        "activities": activities,
        "summary": summary_sentences
    }
    english_raw = eng_chain.invoke({"messages": json.dumps(translation_targets, ensure_ascii=False)})
    english_clean = clean_json_block(english_raw)

    try:
        english = json.loads(english_clean)
        eng_tags = english.get("eng_tags", [])
        eng_activities = english.get("eng_activities", [])
        eng_summary_sentences = english.get("eng_summary", [])
    except json.JSONDecodeError:
        print("⚠️ 영어 파싱 실패! 원본 출력:", english_clean)
        eng_tags, eng_activities, eng_summary_sentences = [], [], []

    # Stable Diffusion용 프롬프트 생성
    structured_summary = {
        "subject": ["I"],
        "location": tags,
        "emotion": [],
        "event": activities,
        "imagery": []
    }

    if summary_sentences:
        first_sentence = summary_sentences[0]
        if "햇살" in first_sentence or "sunlight" in first_sentence:
            structured_summary["imagery"].append("sunlight through the window")
        if "기분" in first_sentence or "느낌" in first_sentence or "feeling" in first_sentence:
            structured_summary["emotion"].append("calm")

    sd_prompt_raw = sd_chain.invoke({"structured_summary": json.dumps(structured_summary, ensure_ascii=False)})
    sd_prompt_clean = clean_json_block(sd_prompt_raw)
    print("\n🎨 Stable Diffusion Prompt:", sd_prompt_clean)

    emotion = analyze_emotion(summary)
    today_str = datetime.now().strftime("%Y-%m-%d")
    date_obj = datetime.strptime(today_str, "%Y-%m-%d")
    filename = f"{current_user.username}_{today_str}.png"
    image_path = generate_image([sd_prompt_clean], output_path=BASE_DIR+"/image/"+filename)


    saved_diary = chat_crud.create_diary(
        db=db,
        username=current_user.username,
        summary=summary,
        emotion=emotion,
        image_url=image_path,
        date=date_obj,
        conversation = request.conversation
    )

    return {
        "summary": summary,
        "emotion": emotion,
        "imageurl": image_path,
        "date" : date_obj,
        "conversation" : request.conversation
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