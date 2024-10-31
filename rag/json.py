from rag.base import RetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class JSONRetrievalChain(RetrievalChain):
    def __init__(self, source_uri):
        self.source_uri = source_uri
        self.k = 5

    def load_documents(self, source_uris: List[str]):
        docs = []
        for source_uri in source_uris:
            loader = JSONLoader(source_uri)
            docs.extend(loader.load())

        return docs

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)