from rag.base import RetrievalChain
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class JSONRetrievalChain(RetrievalChain):
    def __init__(self, source_uri, docs=None):
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
