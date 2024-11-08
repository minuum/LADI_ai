from pdf_rag.base import RetrievalChain
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import re  # 정규 표현식 모듈 추가
from langchain.schema import Document  # Document 클래스 추가

class PDFRetrievalChain(RetrievalChain):
    def __init__(self, source_uri):
        self.source_uri = source_uri
        self.k = 5

    def load_documents(self, source_uris: List[str]):
        docs = []
        for source_uri in source_uris:
            loader = PDFPlumberLoader(source_uri)
            docs.extend(loader.load())
        docs = self.preprocess_documents(docs)

        return docs

    def preprocess_documents(self, docs):
        # 문서 전처리
        # 이스케이프 문자 제거
        docs = [self.remove_escape_sequences(doc.page_content) for doc in docs]
        return [Document(page_content=doc, metadata={}) for doc in docs]  # Document 객체로 변환

    def remove_escape_sequences(self, text):
        # 정규 표현식을 사용하여 이스케이프 문자 제거
        docs=text.replace("\x07", " ").replace("\n", " ")
        return re.sub(r'\\[xX][0-9a-fA-F]{1,2}|\\[0-7]{1,3}|\\[abfnrtv\\"\'?]', '', docs)

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)