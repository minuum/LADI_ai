from rag.base import RetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional
from operator import itemgetter


class JSONRetrievalChain(RetrievalChain):
    def __init__(self, source_uri: Optional[str] = None, docs=None):
        super().__init__()
        self.source_uri = source_uri
        self.k = 5
        self.docs = docs
    def load_documents(self, source_uris: List[str]):
        if self.source_uri == ["pass"]:
            return self.source_uri
        if self.docs is None:
            self.docs = []
            for source_uri in source_uris:
                loader = JSONLoader(source_uri)
                self.docs.extend(loader.load())
        return self.docs

    def create_text_splitter(self):
        if self.source_uri == ["pass"]:
            return self.source_uri
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def create_chain(self, cache_mode: str, local_db: str, category: str, mode: str, func: str):
        """체인을 생성하고 초기화합니다."""
        if func == "makeaction":
            print("makeaction 체인 생성 중")
            docs = self.load_documents(self.source_uri)
            text_splitter = self.create_text_splitter()
            split_docs = self.split_documents(docs, text_splitter, local_db=local_db)
            self.vectorstore = self.create_vectorstore(split_docs, cache_mode=cache_mode, local_db=local_db)
            self.retriever = self.create_retriever(split_docs, category=category, mode=mode)
            model = self.create_model()
            prompt = self.create_prompt(prompt_name=func)

            # structured output을 위한 체인 구성
            self.chain = (
                {
                    "context": self.retriever,
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

        elif func == "makeroutine":
            print("makeroutine 체인 생성 중")
            docs = self.load_documents(self.source_uri)
            text_splitter = self.create_text_splitter()
            split_docs = self.split_documents(docs, text_splitter, local_db=local_db)
            self.vectorstore = self.create_vectorstore(split_docs, cache_mode=cache_mode, local_db=local_db)
            self.retriever = self.create_retriever(split_docs, category=category, mode=mode)
            model = self.create_model(func=func)
            prompt = self.create_prompt(prompt_name=func)

            # structured output을 위한 체인 구성
            self.chain = (
                {
                    "context": self.retriever,
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
