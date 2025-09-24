from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from typing import List

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str, ollama_host: str):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_host
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, docs: List[Document]):
        self.vector_store.add_documents(docs)
        
    def similarity_search(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)