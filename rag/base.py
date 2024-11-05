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
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from typing import Any, Callable, Dict, Iterable, List, Optional
from operator import itemgetter
import numpy as np
import pickle
import os

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever

try:
    from kiwipiepy import Kiwi
except ImportError:
    raise ImportError(
        "Could not import kiwipiepy, please install with `pip install kiwipiepy`."
    )

kiwi_tokenizer = Kiwi()

def kiwi_preprocessing_func(text: Union[str, List[str]]) -> List[str]:
    tokens = []
    try:
        if isinstance(text, list):
            for item in text:
                result = kiwi_tokenizer.tokenize(str(item))
                if isinstance(result, list):
                    # 각 토큰의 form 속성 접근
                    tokens.extend([token.form for token in result if hasattr(token, 'form')])
                else:
                    # 단일 토큰인 경우
                    tokens.append(result.form if hasattr(result, 'form') else str(result))
        else:
            # 문자열인 경우 직접 토큰화
            result = kiwi_tokenizer.tokenize(str(text))
            if isinstance(result, list):
                tokens = [token.form for token in result if hasattr(token, 'form')]
            else:
                tokens = [result.form if hasattr(result, 'form') else str(result)]
    except Exception as e:
        print(f"토큰화 중 오류 발생: {e}")
        print(f"입력 텍스트 타입: {type(text)}")
        print(f"입력 텍스트 내용: {text}")
        # 오류 발생 시 기본 토큰화 방식 사용
        return str(text).split()
    
    return tokens

def default_preprocessing_func(text: str) -> List[str]:
    return text.split()

from tqdm import tqdm  # tqdm 추가

class KiwiBM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = kiwi_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = kiwi_preprocessing_func,
        **kwargs: Any,
    ) -> "KiwiBM25Retriever":
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install rank_bm25`."
            )

        texts_processed = [preprocess_func(t) for t in tqdm(texts, desc="텍스트 처리 중")]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = kiwi_preprocessing_func,
        **kwargs: Any,
    ) -> "KiwiBM25Retriever":
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def save_local(self, folder_path: str, index_name: str) -> None:
        """벡터라이저와 문서를 각각 별도의 파일로 저장합니다."""
        import time
        start_time = time.time()  # 시간 측정 시작
        os.makedirs(folder_path, exist_ok=True)
        
        # 벡터라이저 저장
        with open(os.path.join(folder_path, f"{index_name}_vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # 문서 데이터 저장
        with open(os.path.join(folder_path, f"{index_name}_docs.pkl"), 'wb') as f:
            pickle.dump(self.docs, f)

        end_time = time.time()  # 시간 측정 종료
        print(f"save_local 시간: {end_time - start_time:.2f}초")  # 시간 출력

    @classmethod
    def load_local(cls, folder_path: str, index_name: str, **kwargs) -> "KiwiBM25Retriever":
        """저장된 벡터라이저와 문서를 각각 불러옵니다."""
        import time
        start_time = time.time()  # 시간 측정 시작
        
        # 벡터라이저 로드
        with open(os.path.join(folder_path, f"{index_name}_vectorizer.pkl"), 'rb') as f:
            vectorizer = pickle.load(f)
        
        # 문서 데이터 로드
        with open(os.path.join(folder_path, f"{index_name}_docs.pkl"), 'rb') as f:
            docs = pickle.load(f)
        
        end_time = time.time()  # 시간 측정 종료
        print(f"load_local 시간: {end_time - start_time:.2f}초")  # 시간 출력
        
        return cls(vectorizer=vectorizer, docs=docs, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        return return_docs

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def argsort(seq, reverse):
        return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

    def search_with_score(self, query: str, top_k=None):
        import time
        start_time = time.time()  # 시간 측정 시작

        normalized_score = self.softmax(
            self.vectorizer.get_scores(self.preprocess_func(query))
        )

        if top_k is None:
            top_k = self.k

        score_indexes = self.argsort(normalized_score, True)

        docs_with_scores = []
        for i, doc in tqdm(enumerate(self.docs), desc="문서 점수 계산 중"):
            document = Document(
                page_content=doc.page_content, metadata={"score": normalized_score[i]}
            )
            docs_with_scores.append(document)

        score_indexes = score_indexes[:top_k]
        getter = itemgetter(*score_indexes)
        selected_elements = getter(docs_with_scores)

        end_time = time.time()  # 시간 측정 종료
        print(f"search_with_score 시간: {end_time - start_time:.2f}초")  # 시간 출력

        return selected_elements

from pydantic import BaseModel, Field
from typing import List, Optional

#==========================================================
# 행동 정보
class Action(BaseModel):
    """행동 정보"""
    action: str = Field(description="행동 이름")

class ActionResponse(BaseModel):
    """행동 정보"""
    actions: List[Action] = Field(description="추천 행동 리스트")
    category: str = Field(description="카테고리")
    goal: str = Field(description="목표")


#==========================================================
# 루틴 정보

class RoutineSubStep(BaseModel):
    """루틴의 개별 하위 단계"""
    emoji: str = Field(default="", description="이모지 표현, 이모지만!!!!!!!!!!!!")
    routine: str = Field(default="", description="20자 이내의 직관적인 루틴 설명")
    secondDuration: int = Field(default=0, description="단계의 예상 소요 시간 (초 단위)")


class RoutineResponse(BaseModel):
    """AI 코치의 맞춤형 루틴 응답"""
    subRoutine: List[RoutineSubStep] = Field(description="루틴의 각 단계 정보")

#==========================================================
# 체인 정보

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
    def split_documents(self, docs, text_splitter=None,local_db="./cache/"):
        """
        문서를 처리합니다. 
        대화 데이터의 경우 이미 적절한 단위로 구성되어 있으므로,
        단순히 Document 객체로 변환만 수행합니다.
        """
        if local_db == "./cached_healthcare/":
            documents=[]
            for data in docs:
                doc=Document(page_content=data['content'], 
                             metadata=data['metadata'])
                documents.append(doc)
            return documents  # 전처리 없이 그대로 반환
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


    def create_vectorstore(self, split_docs, cache_mode='load', local_db="./cache/",mode='dense'):
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

    

    def create_retriever(self, split_docs, category=None, mode='dense', cache_dir="cache"):
        if mode == 'dense':
            dense_retriever = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": self.k, "filter": {"category": category}}
            )
            return dense_retriever
        elif mode == 'kiwi':
            cache_path = os.path.join(cache_dir, f"kiwi_bm25_{category}")
            print(f"kiwi_bm25_{category} 캐시 경로: {cache_path}")
            # 캐시된 인덱스가 있는지 확인
            if os.path.exists(f"{cache_path}_vectorizer.pkl") and os.path.exists(f"{cache_path}_docs.pkl"):
                try:
                    print(f"{cache_path} 로드 중")
                    return KiwiBM25Retriever.load_local(cache_dir, f"kiwi_bm25_{category}")
                except Exception as e:
                    print(f"캐시된 인덱스 로드 실패: {e}")
            
            # 캐시가 없거나 로드 실패시 새로 생성
            kiwi = KiwiBM25Retriever.from_documents(documents=split_docs)
            
            # 새로 생성한 인덱스 저장
            os.makedirs(cache_dir, exist_ok=True)
            try:
                print(f"{cache_path} 저장 중")
                kiwi.save_local(cache_dir, f"kiwi_bm25_{category}")
            except Exception as e:
                print(f"인덱스 저장 실패: {e}")
                
            return kiwi
        else:
            raise ValueError("지원하지 않는 모드입니다. 'dense' 또는 'kiwi'를 선택하세요.")

    def create_model(self,func="makeaction"):
        if func == "makeaction":
            model = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0)
            return model.with_structured_output(ActionResponse)
        elif func == "makeroutine":
            model = ChatOpenAI(model_name="gpt-4o-2024-08-06", temperature=0)
            return model.with_structured_output(RoutineResponse)

    def create_prompt(self,prompt_name="makeaction"):
        return hub.pull(f"minuum/ladi-{prompt_name}")
    
    @staticmethod
    def format_docs(docs):
        return "\n".join(docs)

    def create_chain(self, cache_mode='load', local_db="./cache/", category=None, mode='dense',func="makeaction"):

        if func == "makeaction":
            docs = self.load_documents(self.source_uri)
            text_splitter = self.create_text_splitter()
            split_docs = self.split_documents(docs, text_splitter,local_db=local_db)
            self.vectorstore = self.create_vectorstore(split_docs, cache_mode=cache_mode, local_db=local_db)
            self.retriever = self.create_retriever(split_docs, category=category, mode=mode)
            model = self.create_model()
            prompt = self.create_prompt(prompt_name=func)
        


        # structured output을 위한 체인 구성
            self.chain = (
                {
                    "context": itemgetter("context"),
                    "question": itemgetter("goal"),
                    "user_id": itemgetter("user_id"),
                    "name": itemgetter("name"),
                    "age": itemgetter("age"),
                    "gender": itemgetter("gender"),
                    "job": itemgetter("job"),
                    "weight": itemgetter("weight"),
                    "height": itemgetter("height"),
                    "workout_frequency": itemgetter("workout_frequency"),
                    "workout_location": itemgetter("workout_location"),
                    "category": itemgetter("category"),
                    "goal": itemgetter("goal")
                }
                | prompt
                | model
            )
            return self

        if func == "makeroutine":
            docs = self.load_documents(self.source_uri)
            text_splitter = self.create_text_splitter()
            split_docs = self.split_documents(docs, text_splitter,local_db=local_db)
            self.vectorstore = self.create_vectorstore(split_docs, cache_mode=cache_mode, local_db=local_db)
            self.retriever = self.create_retriever(split_docs, category=category, mode=mode)
            model = self.create_model(func=func)
            prompt = self.create_prompt(prompt_name=func)
        


            # structured output을 위한 체인 구성
            self.chain = (
                {
                    "context": itemgetter("context"),
                    "question": itemgetter("goal"),
                    "user_id": itemgetter("user_id"),
                    "name": itemgetter("name"),
                    "age": itemgetter("age"),
                    "gender": itemgetter("gender"),
                    "job": itemgetter("job"),
                    "weight": itemgetter("weight"),
                    "height": itemgetter("height"),
                    "workout_frequency": itemgetter("workout_frequency"),
                    "workout_location": itemgetter("workout_location"),
                    "category": itemgetter("category"),
                    "goal": itemgetter("goal"),
                    "selected_actions": itemgetter("selected_actions"),
                    "start_time": itemgetter("start_time"),
                    "repeat_days": itemgetter("repeat_days"),
                }
                | prompt
                | model
            )
        return self
    
