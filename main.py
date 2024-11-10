from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from rag.base import RetrievalChain
from rag.json import JSONRetrievalChain
import requests
import uvicorn

app = FastAPI()

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

# 상위 3개의 문서 선택
compressor = CrossEncoderReranker(model=model, top_n=3)

class UserInfo(BaseModel):
    user_id: int
    name: str
    age: int
    height: float
    weight: float
    workout_frequency: str
    workout_location: str
    gender: str
    job: str

class HabitInfo(BaseModel):
    category: str
    goal: str

class Action(BaseModel):
    action: str

class RoutineInfo(BaseModel):
    selected_actions: List[str]
    start_time: str
    repeat_days: List[str]

class ActionResponse(BaseModel):
    actions: List[Action]

# RAG 체인을 애플리케이션 시작 시 한 번만 초기화
def initialize_rag_chain(func_name):
    chain = JSONRetrievalChain(source_uri=["pass"])  # source_uri는 선택적 파라미터로 변경됨
    chain = chain.create_chain(
        cache_mode='load',
        local_db="./cached_healthcare/",
        category="의료",
        mode='hybrid',
        func=func_name
    )
    retriever = chain.retriever
    rag = chain.chain
    return chain, retriever, rag

# 각 기능에 대한 RAG 체인 초기화
rag_chain_makeaction, rag_retriever_makeaction, rag_makeaction = initialize_rag_chain('makeaction')
rag_chain_makeroutine, rag_retriever_makeroutine, rag_makeroutine = initialize_rag_chain('makeroutine')

@app.post("/make_action", response_model=dict)
async def make_action(
    user_info: UserInfo,
    habit_info: HabitInfo
):
    try:
        # make_action_query 구성
        make_action_query = {
            "context": None,  # context는 chain 내부에서 처리하므로 None으로 설정
            "question": habit_info.goal,
            "user_id": user_info.user_id,
            "name": user_info.name,
            "age": user_info.age,
            "height": user_info.height,
            "weight": user_info.weight,
            "workout_frequency": user_info.workout_frequency,
            "workout_location": user_info.workout_location,
            "gender": user_info.gender,
            "job": user_info.job,
            "category": habit_info.category,
            "goal": habit_info.goal
        }

        # rag_makeaction을 통해 결과 얻기
        result = rag_makeaction.invoke(make_action_query)

        # 결과를 actions_dict 형태로 변환
        actions_dict = {
            "actions": [action.action for action in result.actions]
        }
        # 응답 데이터 생성
        response_data = {**make_action_query, **actions_dict}

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/make_routine", response_model=dict)
async def make_routine(
    make_action_result_query: dict,
    routine_info: RoutineInfo
):
    try:
        # make_routine_query 구성
        make_routine_query = {
            "context": None,  # context는 chain 내부에서 처리하므로 None으로 설정
            "question": make_action_result_query["goal"],
            "user_id": make_action_result_query["user_id"],
            "name": make_action_result_query["name"],
            "age": make_action_result_query["age"],
            "gender": make_action_result_query["gender"],
            "job": make_action_result_query["job"],
            "weight": make_action_result_query["weight"],
            "height": make_action_result_query["height"],
            "workout_frequency": make_action_result_query["workout_frequency"],
            "workout_location": make_action_result_query["workout_location"],
            "category": make_action_result_query["category"],
            "goal": make_action_result_query["goal"],
            "selected_actions": routine_info.selected_actions,
            "start_time": routine_info.start_time,
            "repeat_days": routine_info.repeat_days
        }

        # rag_makeroutine을 통해 결과 얻기
        result = rag_makeroutine.invoke(make_routine_query)

        subroutine_dict = {
            "subroutines": [
                {
                    "emoji": sub_step.emoji,
                    "routine": sub_step.routine,
                    "secondDuration": sub_step.secondDuration
                } for sub_step in result.subRoutine
            ]
        }
        return subroutine_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)