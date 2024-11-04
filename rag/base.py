from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import UpstageEmbeddings
from langchain_openai import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from langchain_core.output_parsers import StrOutputParser
from abc import ABC, abstractmethod
from operator import itemgetter
from langchain.schema import Document
from tqdm import tqdm
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class RoutineStep(BaseModel):
    """루틴의 세부 단계"""
    title: str = Field(description="루틴 단계의 제목")
    steps: List[str] = Field(description="세부 실천 단계들")
    duration: str = Field(description="예상 소요 시간")
    difficulty: str = Field(description="난이도 (상/중/하)")
    category: str = Field(description="카테고리 (운동/식단/학습/취미/생활/기타)")

class RoutineResponse(BaseModel):
    """AI 코치의 맞춤형 루틴 응답"""
    greeting: str = Field(description="사용자를 위한 개인화된 인사말")
    routines: List[RoutineStep] = Field(description="추천된 루틴 단계들")
    tips: Optional[List[str]] = Field(description="실천을 위한 조언이나 팁", default=None)

class RetrievalChain(ABC):
    def __init__(self):
        self.source_uri = None
        self.k = 20

    @abstractmethod
    def load_documents(self, source_uris):
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """text splitter를 생성합니다."""
        pass

    # def split_documents(self, docs, text_splitter):
    #     """text splitter를 사용하여 문서를 분할합니다."""
    #     return text_splitter.split_documents(docs)
    def split_documents(self, docs, text_splitter=None):
        """
        문서를 처리합니다. 
        대화 데이터의 경우 이미 적절한 단위로 구성되어 있으므로,
        단순히 Document 객체로 변환만 수행합니다.
        """
        documents = []
        for data in docs:
            # 대화 내용을 하나의 문자열로 결합
            conversation = "\n".join([utterance['text'] for utterance in data['content']])
            
            # Document 객체 생성
            doc = Document(
                page_content=conversation,
                metadata=data['metadata']
            )
            documents.append(doc)
        return documents
    
    def create_embedding(self):
        
        return UpstageEmbeddings(model="solar-embedding-1-large")


    def create_vectorstore(self, split_docs, cache_mode='load', local_db="./cache/"):
        if cache_mode == 'store':
            print("로컬에 새로운 벡터 저장소를 생성중")
            fs = LocalFileStore(local_db)
            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
                self.create_embedding(), fs, namespace=self.create_embedding().model
            )
            vectorstore = FAISS.from_documents(
                documents=split_docs, embedding=cached_embeddings
            )
            vectorstore.save_local(local_db)
 
            return vectorstore
        elif cache_mode == 'load':
            print("로컬에서 벡터 저장소 로드 중")
            return FAISS.load_local(local_db, self.create_embedding(), allow_dangerous_deserialization=True)
        else:
            print("벡터 저장소를 생성중")
            return FAISS.from_documents(
                documents=split_docs, embedding=self.create_embedding()
            )

    

    def create_retriever(self, split_docs, category=None, mode='dense'):
        # 입력된 mode에 따라 리트리버를 생성합니다.
        if mode == 'dense':
            dense_retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": self.k, "filter": {"category": category}}
            )
            return dense_retriever
        elif mode == 'kiwi':
            from langchain_teddynote.retrievers import KiwiBM25Retriever
            # split_docs가 문서 리스트인지 확인
            if isinstance(split_docs, list):
                kiwi = KiwiBM25Retriever.from_documents(documents=split_docs)
            else:
                raise TypeError("split_docs는 문서 리스트여야 합니다.")
            return kiwi
        else:
            raise ValueError("지원하지 않는 모드입니다. 'dense' 또는 'kiwi'를 선택하세요.")

    def create_model(self):
        model = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0)
        return model.with_structured_output(RoutineResponse)

    def create_prompt(self):
        return hub.pull("minuum/ladi-common")
    
    @staticmethod
    def format_docs(docs):
        return "\n".join(docs)

    def create_chain(self, cache_mode='load', local_db="./cache/", category=None, mode='dense'):
        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs, cache_mode=cache_mode, local_db=local_db)
        self.retriever = self.create_retriever(split_docs, category=category, mode=mode)
        model = self.create_model()
        prompt = self.create_prompt()
        
        # structured output을 위한 체인 구성
        self.chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "USER_ID": itemgetter("USER_ID"),
                "NAME": itemgetter("NAME"),
                "age": itemgetter("age"),
                "GOAL": itemgetter("GOAL"),  # 전반적인 목표
                "gender": itemgetter("gender"),
                "JOB": itemgetter("JOB"),
                "weight": itemgetter("weight"),
                "height": itemgetter("height"),
                "workout_frequency": itemgetter("workout_frequency"),
                "workout_location": itemgetter("workout_location"),
                "category": itemgetter("category"),
                "goal": itemgetter("goal"),
                "selected_tasks": itemgetter("selected_tasks"),
                "start_time": itemgetter("start_time"),
                "repeat_days": itemgetter("repeat_days"),
                "notification": itemgetter("notification")
            }
            | prompt
            | model
        )
        return self
    
