from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def extract_text(self, docs: List[Dict], keys: List[str], metadata_keys: List[str]) -> List[Dict]:
        """
        Return a list of dicts with 'text' and 'metadata' per document,
        preserving each key separately.
        """
        all_doc_combined = []
        for doc in docs:
            combined_doc = ""
            for key in keys:
                if key in doc and doc[key]:
                    combined_doc += f'\n{key}: {doc[key]}\n'
            metadata = {k: doc.get(k) for k in metadata_keys if k in doc}
            all_doc_combined.append({
                "Data": combined_doc,
                "metadata": metadata
            })
        return all_doc_combined
    
    def split_documents(self, docs_with_metadata: List[Dict]) -> List[Document]:
        """
        Split texts into chunks, preserving metadata for each chunk.
        """
        all_chunks = []
        for item in docs_with_metadata:
            text = item["Data"]
            metadata = item.get("metadata", {})
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(Document(page_content=chunk, metadata=metadata))
        return all_chunks
