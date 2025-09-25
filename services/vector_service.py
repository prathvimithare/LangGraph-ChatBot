from services.opensearch import OpenSearchClient
from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore
from config import Config
from typing import List, Tuple
from langchain.schema import Document

class VectorService:
    def __init__(self, index_name: str, collection_name: str, text_keys: List[str], metadata_keys: List[str]):
        self.index_name = index_name
        self.collection_name = collection_name
        self.text_keys = text_keys
        self.metadata_keys = metadata_keys

        self.client = OpenSearchClient(host=Config.OPENSEARCH_URL, index_name=index_name)
        self.processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory="ChromaDB",
            ollama_host=Config.OLLAMA_HOST
        )
    
    def index_documents(self):
        """Fetch documents, process and chunk them, then add to vector store."""
        docs = self.client.get_all_documents()
        print(f"Fetched {len(docs)} documents from {self.index_name}")

        texts = self.processor.extract_text(docs, keys=self.text_keys, metadata_keys=self.metadata_keys)
        chunks = self.processor.split_documents(texts)

        if chunks:
            # self.vector_store.add_documents(chunks)
            print(f"Added {len(chunks)} chunks to vector store collection {self.collection_name}")
        else:
            print("No chunks to add.")

    # def build_retriever(self, k: int = 5):
    #     """Return a Chroma retriever for querying."""
    #     retriever = self.vector_store.vector_store.as_retriever(
    #         search_type="similarity",
    #         search_kwargs={"k": k}
    #     )
    #     return retriever
    
    def build_retriever(self, k: int = 5, threshold: float = 0.0):
        """
        Returns a custom retriever function that filters by similarity score.
        """
        vector_store = self.vector_store

        class Retriever:
            def __init__(self, vector_store: VectorStore, k: int, threshold: float):
                self.vector_store = vector_store
                self.k = k
                self.threshold = threshold

            def invoke(self, query: str) -> List[Document]:
                # Get documents with similarity scores
                results: List[Tuple[Document, float]] = self.vector_store.vector_store.similarity_search_with_score(query, k=self.k)
                # Filter by threshold
                filtered_docs = [doc for doc, score in results if score >= self.threshold]
                scores = [score for doc, score in results if score >= self.threshold]
                return filtered_docs, scores

        return Retriever(vector_store, k, threshold)
